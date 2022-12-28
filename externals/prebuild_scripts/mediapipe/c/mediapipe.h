#ifndef MEDIAPIPE_H
#define MEDIAPIPE_H

#include <stdint.h>

#if _WIN32
    #ifdef COMPILING_DLL
        #define MEDIAPIPE_API __declspec(dllexport)
    #else
        #define MEDIAPIPE_API __declspec(dllimport)
    #endif
#else
    #define MEDIAPIPE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mediapipe_instance mediapipe_instance;
typedef struct mediapipe_packet mediapipe_packet;

typedef struct {
    const uint8_t* data;
    int width;
    int height;
    int format;
} mediapipe_image;

MEDIAPIPE_API mediapipe_instance* mediapipe_create_instance(const char* graph, const char* input_stream);
MEDIAPIPE_API bool mediapipe_add_output_stream(mediapipe_instance* instance, const char* output_stream);
MEDIAPIPE_API bool mediapipe_start(mediapipe_instance* instance);
MEDIAPIPE_API bool mediapipe_process(mediapipe_instance* instance, mediapipe_image image);
MEDIAPIPE_API bool mediapipe_wait_until_idle(mediapipe_instance* instance);
MEDIAPIPE_API mediapipe_packet* mediapipe_poll_packet(mediapipe_instance* instance, const char* name);
MEDIAPIPE_API void mediapipe_destroy_packet(mediapipe_packet* packet);
MEDIAPIPE_API void mediapipe_close_instance(mediapipe_instance* instance);
MEDIAPIPE_API void mediapipe_set_resource_dir(const char* dir);

MEDIAPIPE_API void mediapipe_read_packet_image(mediapipe_packet* packet, uint8_t* out_data);

#ifdef __cplusplus
}
#endif

#endif