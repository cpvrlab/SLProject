#include "mediapipe.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/tool/options_util.h"

#include "mediapipe/calculators/util/thresholding_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "google/protobuf/util/json_util.h"

#include <string>
#include <cstring>
#include <variant>
#include <cassert>
#include <fstream>
#include <iostream>

ABSL_DECLARE_FLAG(std::string, resource_root_dir);
static absl::Status last_error;

struct mediapipe_node_option {
    const char* node;
    const char* option;
    std::variant<float, double> value;
};

struct mediapipe_instance_builder {
    const char* graph_filename;
    const char* input_stream;
    std::vector<mediapipe_node_option> options;
    std::map<std::string, mediapipe::Packet> side_packets;
};

struct mediapipe_instance {
    mediapipe::CalculatorGraph graph;
    std::string input_stream;
    size_t frame_timestamp;
};

struct mediapipe_poller {
    mediapipe::OutputStreamPoller poller;
};

struct mediapipe_packet {
    mediapipe::Packet packet;
};

template<typename List, typename Landmark>
static mediapipe_multi_face_landmark_list* get_multi_face_landmarks(mediapipe_packet* packet) {
    const auto& mp_data = packet->packet.template Get<std::vector<List>>();

    auto* lists = new mediapipe_landmark_list[mp_data.size()];

    for (int i = 0; i < mp_data.size(); i++) {
        const List& mp_list = mp_data[i];
        auto* list = new mediapipe_landmark[mp_list.landmark_size()];

        for (int j = 0; j < mp_list.landmark_size(); j++) {
            const Landmark& mp_landmark = mp_list.landmark(j);
            list[j] = mediapipe_landmark {
                mp_landmark.x(),
                mp_landmark.y(),
                mp_landmark.z()
            };
        }

        lists[i] = mediapipe_landmark_list {
            list,
            (int) mp_list.landmark_size()
        };
    }

    return new mediapipe_multi_face_landmark_list {
        lists,
        (int) mp_data.size()
    };
}

extern "C" {

MEDIAPIPE_API mediapipe_instance_builder* mediapipe_create_instance_builder(const char* graph_filename, const char* input_stream) {
    return new mediapipe_instance_builder { graph_filename, input_stream, {} };
}

MEDIAPIPE_API void mediapipe_add_option_float(mediapipe_instance_builder* instance_builder, const char* node, const char* option, float value) {
    instance_builder->options.push_back({ node, option, value });
}

MEDIAPIPE_API void mediapipe_add_option_double(mediapipe_instance_builder* instance_builder, const char* node, const char* option, double value) {
    instance_builder->options.push_back({ node, option, value });
}

MEDIAPIPE_API void mediapipe_add_side_packet(mediapipe_instance_builder* instance_builder, const char* name, mediapipe_packet* packet) {
    instance_builder->side_packets.insert({name, packet->packet});
    mediapipe_destroy_packet(packet);
}

MEDIAPIPE_API mediapipe_instance* mediapipe_create_instance(mediapipe_instance_builder* builder) {
    mediapipe::CalculatorGraphConfig config;
    
    std::ifstream stream(builder->graph_filename, std::ios::binary | std::ios::ate);
    size_t size = stream.tellg();
    stream.seekg(0, std::ios::beg);

    char* memory = new char[size];
    stream.read(memory, size);
    config.ParseFromArray(memory, size);
    delete[] memory;
    
    mediapipe::ValidatedGraphConfig validated_config;
    validated_config.Initialize(config);
    mediapipe::CalculatorGraphConfig canonical_config = validated_config.Config();

    for (const mediapipe_node_option& option : builder->options) {
        for (auto& node : *canonical_config.mutable_node()) {
            if (node.name() != option.node) {
                continue;
            }

            google::protobuf::Message* ext;

            if (node.calculator() == "ThresholdingCalculator")
                ext = node.mutable_options()->MutableExtension(mediapipe::ThresholdingCalculatorOptions::ext);
            else if (node.calculator() == "TensorsToDetectionsCalculator")
                ext = node.mutable_options()->MutableExtension(mediapipe::TensorsToDetectionsCalculatorOptions::ext);
            else {
                assert(!"Unknown node calculator");
                return nullptr;
            }

            auto* descriptor = ext->GetDescriptor();
            auto* reflection = ext->GetReflection();
            auto* field_descriptor = descriptor->FindFieldByName(option.option);
            
            switch (option.value.index()) {
                case 0: reflection->SetFloat(ext, field_descriptor, std::get<0>(option.value)); break;
                case 1: reflection->SetDouble(ext, field_descriptor, std::get<1>(option.value)); break;
            }

            switch (option.value.index()) {
                case 0: std::cout << reflection->GetFloat(*ext, field_descriptor) << std::endl; break;
                case 1: std::cout << reflection->GetDouble(*ext, field_descriptor) << std::endl; break;
            }
        }
    }

    google::protobuf::util::JsonPrintOptions json_options;
    json_options.add_whitespace = true;

    std::string str;
    google::protobuf::util::MessageToJsonString(canonical_config, &str, json_options);
    std::cout << str << std::endl;

    auto* instance = new mediapipe_instance;
    absl::Status result = instance->graph.Initialize(canonical_config, builder->side_packets);
    if (!result.ok()) {
        last_error = result;
        return nullptr;
    }

    instance->input_stream = builder->input_stream;
    instance->frame_timestamp = 0;

    delete builder;
    return instance;
}

MEDIAPIPE_API mediapipe_poller* mediapipe_create_poller(mediapipe_instance* instance, const char* output_stream) {
    absl::StatusOr<mediapipe::OutputStreamPoller> result = instance->graph.AddOutputStreamPoller(output_stream);
    if (!result.ok()) {
        last_error = result.status();
        return nullptr;
    }

    return new mediapipe_poller { 
        std::move(*result)
    };
}

MEDIAPIPE_API bool mediapipe_start(mediapipe_instance* instance) {
    absl::Status result = instance->graph.StartRun({});

    if (!result.ok()) {
        last_error = result;
        return false;
    }

    return true;
}

MEDIAPIPE_API bool mediapipe_process(mediapipe_instance* instance, mediapipe_image image) {
    auto mp_frame = std::make_unique<mediapipe::ImageFrame>();
    auto mp_format = static_cast<mediapipe::ImageFormat::Format>(image.format);
    uint32_t mp_alignment_boundary = mediapipe::ImageFrame::kDefaultAlignmentBoundary;
    mp_frame->CopyPixelData(mp_format, image.width, image.height, image.data, mp_alignment_boundary);

    mediapipe::Timestamp mp_timestamp(instance->frame_timestamp++);

    mediapipe::Packet packet = mediapipe::Adopt(mp_frame.release()).At(mp_timestamp);
    auto result = instance->graph.AddPacketToInputStream(instance->input_stream, packet);
    
    if (!result.ok()) {
        last_error = result;
        return false;
    }

    return true;
}

MEDIAPIPE_API bool mediapipe_wait_until_idle(mediapipe_instance* instance) {
    absl::Status result = instance->graph.WaitUntilIdle();

    if (!result.ok()) {
        last_error = result;
        return false;
    }

    return true;
}

MEDIAPIPE_API int mediapipe_get_queue_size(mediapipe_poller* poller) {
    return poller->poller.QueueSize();
}

MEDIAPIPE_API void mediapipe_destroy_poller(mediapipe_poller* poller) {
    delete poller;
}

MEDIAPIPE_API bool mediapipe_destroy_instance(mediapipe_instance* instance) {
    absl::Status result = instance->graph.CloseInputStream(instance->input_stream);
    if (!result.ok()) {
        last_error = result;
        return false;
    }

    result = instance->graph.WaitUntilDone();
    if (!result.ok()) {
        last_error = result;
        return false;
    }
    
    delete instance;
    return true;
}

MEDIAPIPE_API void mediapipe_set_resource_dir(const char* dir) {
    absl::SetFlag(&FLAGS_resource_root_dir, dir);
}

MEDIAPIPE_API mediapipe_packet* mediapipe_create_packet_int(int value) {
    return new mediapipe_packet {
      .packet = mediapipe::MakePacket<int>(value)
    };
}

MEDIAPIPE_API mediapipe_packet* mediapipe_create_packet_float(float value) {
    return new mediapipe_packet {
      .packet = mediapipe::MakePacket<float>(value)
    };
}

MEDIAPIPE_API mediapipe_packet* mediapipe_create_packet_bool(bool value) {
    return new mediapipe_packet {
      .packet = mediapipe::MakePacket<bool>(value)
    };
}

MEDIAPIPE_API mediapipe_packet* mediapipe_poll_packet(mediapipe_poller* poller) {
    auto* packet = new mediapipe_packet;
    poller->poller.Next(&packet->packet);
    return packet;
}

MEDIAPIPE_API void mediapipe_destroy_packet(mediapipe_packet* packet) {
    delete packet;
}

MEDIAPIPE_API size_t mediapipe_get_packet_type_len(mediapipe_packet* packet) {
    mediapipe::TypeId type = packet->packet.GetTypeId();
    return type.name().size();
}

MEDIAPIPE_API void mediapipe_get_packet_type(mediapipe_packet* packet, char* buffer) {
    mediapipe::TypeId type = packet->packet.GetTypeId();
    std::strcpy(buffer, type.name().c_str());
}

MEDIAPIPE_API void mediapipe_copy_packet_image(mediapipe_packet* packet, uint8_t* out_data) {
    const auto& mp_frame = packet->packet.Get<mediapipe::ImageFrame>();
    size_t data_size = mp_frame.PixelDataSizeStoredContiguously();
    mp_frame.CopyToBuffer(out_data, data_size);
}

MEDIAPIPE_API mediapipe_multi_face_landmark_list* mediapipe_get_multi_face_landmarks(mediapipe_packet* packet) {
    return get_multi_face_landmarks<mediapipe::LandmarkList, mediapipe::Landmark>(packet);
}

MEDIAPIPE_API mediapipe_multi_face_landmark_list* mediapipe_get_normalized_multi_face_landmarks(mediapipe_packet* packet) {
    return get_multi_face_landmarks<mediapipe::NormalizedLandmarkList, mediapipe::NormalizedLandmark>(packet);
}

MEDIAPIPE_API void mediapipe_destroy_multi_face_landmarks(mediapipe_multi_face_landmark_list* multi_face_landmarks) {
    for (int i = 0; i < multi_face_landmarks->length; i++) {
        delete[] multi_face_landmarks->elements[i].elements;
    }
    
    delete[] multi_face_landmarks->elements;
    delete multi_face_landmarks;
}

MEDIAPIPE_API void mediapipe_print_last_error() {
    std::cout << "[MediaPipe] " << last_error << std::endl;
}

}
