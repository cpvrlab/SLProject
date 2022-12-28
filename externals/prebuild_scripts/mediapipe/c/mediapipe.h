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
typedef struct mediapipe_poller mediapipe_poller;
typedef struct mediapipe_packet mediapipe_packet;

typedef struct {
    const uint8_t* data;
    int width;
    int height;
    int format;
} mediapipe_image;

typedef struct {
    float x;
    float y;
    float z;
} mediapipe_landmark;

typedef struct {
    mediapipe_landmark* elements;
    int length;
} mediapipe_landmark_list;

typedef struct {
    mediapipe_landmark_list* elements;
    int length;
} mediapipe_multi_face_landmark_list;

// https://google.github.io/mediapipe/solutions/hands.html#hand-landmark-model
typedef enum {
    mediapipe_hand_landmark_wrist = 0,
    mediapipe_hand_landmark_thumb_cmc = 1,
    mediapipe_hand_landmark_thumb_mcp = 2,
    mediapipe_hand_landmark_thumb_ip = 3,
    mediapipe_hand_landmark_thumb_tip = 4,
    mediapipe_hand_landmark_index_finger_mcp = 5,
    mediapipe_hand_landmark_index_finger_pip = 6,
    mediapipe_hand_landmark_index_finger_dip = 7,
    mediapipe_hand_landmark_index_finger_tip = 8,
    mediapipe_hand_landmark_middle_finger_mcp = 9,
    mediapipe_hand_landmark_middle_finger_pip = 10,
    mediapipe_hand_landmark_middle_finger_dip = 11,
    mediapipe_hand_landmark_middle_finger_tip = 12,
    mediapipe_hand_landmark_ring_finger_mcp = 13,
    mediapipe_hand_landmark_ring_finger_pip = 14,
    mediapipe_hand_landmark_ring_finger_dip = 15,
    mediapipe_hand_landmark_ring_finger_tip = 16,
    mediapipe_hand_landmark_pinky_mcp = 17,
    mediapipe_hand_landmark_pinky_pip = 18,
    mediapipe_hand_landmark_pinky_dip = 19,
    mediapipe_hand_landmark_pinky_tip = 20
} mediapipe_hand_landmark;

MEDIAPIPE_API mediapipe_instance* mediapipe_create_instance(const char* graph, const char* input_stream);
MEDIAPIPE_API mediapipe_poller* mediapipe_create_poller(mediapipe_instance* instance, const char* output_stream);
MEDIAPIPE_API bool mediapipe_start(mediapipe_instance* instance);
MEDIAPIPE_API bool mediapipe_process(mediapipe_instance* instance, mediapipe_image image);
MEDIAPIPE_API bool mediapipe_wait_until_idle(mediapipe_instance* instance);
MEDIAPIPE_API mediapipe_packet* mediapipe_poll_packet(mediapipe_poller* poller);
MEDIAPIPE_API void mediapipe_destroy_packet(mediapipe_packet* packet);
MEDIAPIPE_API int mediapipe_get_queue_size(mediapipe_poller* poller);
MEDIAPIPE_API void mediapipe_destroy_poller(mediapipe_poller* poller);
MEDIAPIPE_API void mediapipe_close_instance(mediapipe_instance* instance);
MEDIAPIPE_API void mediapipe_set_resource_dir(const char* dir);

MEDIAPIPE_API size_t mediapipe_get_packet_type_len(mediapipe_packet* packet);
MEDIAPIPE_API void mediapipe_get_packet_type(mediapipe_packet* packet, char* buffer);
MEDIAPIPE_API void mediapipe_read_packet_image(mediapipe_packet* packet, uint8_t* out_data);
MEDIAPIPE_API mediapipe_multi_face_landmark_list* mediapipe_get_multi_face_landmarks(mediapipe_packet* packet);
MEDIAPIPE_API void mediapipe_destroy_multi_face_landmarks(mediapipe_multi_face_landmark_list* multi_face_landmarks);

#ifdef __cplusplus
}
#endif

#endif