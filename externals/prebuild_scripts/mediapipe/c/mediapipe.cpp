#include "mediapipe.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include <string>
#include <unordered_map>

ABSL_DECLARE_FLAG(std::string, resource_root_dir);

extern "C" {

struct mediapipe_instance {
    mediapipe::CalculatorGraph graph;
    std::string input_stream;
    std::unordered_map<std::string, mediapipe::OutputStreamPoller> pollers;
    size_t frame_timestamp;
};

struct mediapipe_packet {
    mediapipe::Packet packet;
};

MEDIAPIPE_API mediapipe_instance* mediapipe_create_instance(const char* graph, const char* input_stream) {
    auto* instance = new mediapipe_instance;
    
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph);
    instance->graph.Initialize(config);
    instance->input_stream = input_stream;
    instance->frame_timestamp = 0;

    return instance;
}

MEDIAPIPE_API bool mediapipe_add_output_stream(mediapipe_instance* instance, const char* output_stream) {    
    absl::StatusOr<mediapipe::OutputStreamPoller> result = instance->graph.AddOutputStreamPoller(output_stream);
    if (!result.ok()) {
        return false;
    }

    instance->pollers.emplace(output_stream, std::move(*result));
    return true;
}

MEDIAPIPE_API bool mediapipe_start(mediapipe_instance* instance) {
    return instance->graph.StartRun({}).ok();
}

MEDIAPIPE_API bool mediapipe_process(mediapipe_instance* instance, mediapipe_image image) {
    auto mp_frame = std::make_unique<mediapipe::ImageFrame>();
    auto mp_format = static_cast<mediapipe::ImageFormat::Format>(image.format);
    uint32_t mp_alignment_boundary = mediapipe::ImageFrame::kDefaultAlignmentBoundary;
    mp_frame->CopyPixelData(mp_format, image.width, image.height, image.data, mp_alignment_boundary);

    mediapipe::Timestamp mp_timestamp(instance->frame_timestamp++);

    mediapipe::Packet packet = mediapipe::Adopt(mp_frame.release()).At(mp_timestamp);
    auto result = instance->graph.AddPacketToInputStream(instance->input_stream, packet);
    return result.ok();
}

MEDIAPIPE_API bool mediapipe_wait_until_idle(mediapipe_instance* instance) {
    return instance->graph.WaitUntilIdle().ok();
}

MEDIAPIPE_API mediapipe_packet* mediapipe_poll_packet(mediapipe_instance* instance, const char* name) {
    auto* packet = new mediapipe_packet;
    instance->pollers.at(name).Next(&packet->packet);
    return packet;
}

MEDIAPIPE_API void mediapipe_destroy_packet(mediapipe_packet* packet) {
    delete packet;
}

MEDIAPIPE_API void mediapipe_close_instance(mediapipe_instance* instance) {
    delete instance;
}

MEDIAPIPE_API void mediapipe_set_resource_dir(const char* dir) {
    absl::SetFlag(&FLAGS_resource_root_dir, dir);
}

MEDIAPIPE_API void mediapipe_read_packet_image(mediapipe_packet* packet, uint8_t* out_data) {
    const auto& mp_frame = packet->packet.Get<mediapipe::ImageFrame>();
    size_t data_size = mp_frame.PixelDataSizeStoredContiguously();
    mp_frame.CopyToBuffer(out_data, data_size);
}

}