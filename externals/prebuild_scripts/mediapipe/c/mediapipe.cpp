#include "mediapipe.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"

#include <string>
#include <cstring>
#include <iostream>

ABSL_DECLARE_FLAG(std::string, resource_root_dir);
static absl::Status last_error;

extern "C" {

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

MEDIAPIPE_API mediapipe_instance* mediapipe_create_instance(const char* graph, const char* input_stream) {    
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph);
    
    auto* instance = new mediapipe_instance;
    absl::Status result = instance->graph.Initialize(config);
    if (!result.ok()) {
        last_error = result;
        return nullptr;
    }
    
    instance->input_stream = input_stream;
    instance->frame_timestamp = 0;

    return instance;
}

MEDIAPIPE_API mediapipe_poller* mediapipe_create_poller(mediapipe_instance* instance, const char* output_stream) {
    absl::StatusOr<mediapipe::OutputStreamPoller> result = instance->graph.AddOutputStreamPoller(output_stream);
    if (!result.ok()) {
        last_error = result.status();
        return nullptr;
    }

    return new mediapipe_poller { 
        .poller = std::move(*result)
    };;
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

MEDIAPIPE_API mediapipe_packet* mediapipe_poll_packet(mediapipe_poller* poller) {
    auto* packet = new mediapipe_packet;
    poller->poller.Next(&packet->packet);
    return packet;
}

MEDIAPIPE_API void mediapipe_destroy_packet(mediapipe_packet* packet) {
    delete packet;
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

MEDIAPIPE_API size_t mediapipe_get_packet_type_len(mediapipe_packet* packet) {
    mediapipe::TypeId type = packet->packet.GetTypeId();
    return type.name().size();
}

MEDIAPIPE_API void mediapipe_get_packet_type(mediapipe_packet* packet, char* buffer) {
    mediapipe::TypeId type = packet->packet.GetTypeId();
    std::strcpy(buffer, type.name().c_str());
}

MEDIAPIPE_API void mediapipe_read_packet_image(mediapipe_packet* packet, uint8_t* out_data) {
	const auto& mp_frame = packet->packet.Get<mediapipe::ImageFrame>();
	size_t data_size = mp_frame.PixelDataSizeStoredContiguously();
    mp_frame.CopyToBuffer(out_data, data_size);
}

MEDIAPIPE_API mediapipe_multi_face_landmark_list* mediapipe_get_multi_face_landmarks(mediapipe_packet* packet) {
    const auto& mp_data = packet->packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

    auto* lists = new mediapipe_landmark_list[mp_data.size()];
    
    for (int i = 0; i < mp_data.size(); i++) {
        const mediapipe::NormalizedLandmarkList& mp_list = mp_data[i];
        auto* list = new mediapipe_landmark[mp_list.landmark_size()];
        
        for (int j = 0; j < mp_list.landmark_size(); j++) {
            const mediapipe::NormalizedLandmark& mp_landmark = mp_list.landmark(j);
            list[j] = mediapipe_landmark {
                .x = mp_landmark.x(),
                .y = mp_landmark.y(),
                .z = mp_landmark.z()
            };
        }

        lists[i] = mediapipe_landmark_list {
            .elements = list,
            .length = (int) mp_list.landmark_size()
        };
    }

    return new mediapipe_multi_face_landmark_list {
        .elements = lists,
        .length = (int) mp_data.size()
    };
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