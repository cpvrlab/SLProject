#ifndef MEDIAPIPE_H
#define MEDIAPIPE_H

#include <stdint.h>
#include <stddef.h>

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

/// Object for configuring an instance before creating it.
typedef struct mediapipe_instance_builder mediapipe_instance_builder;

/// Contains the MediaPipe graph as well as some extra information.
/// Data in MediaPipe flows between node of a graph to perform computations.
/// The graph takes some packets via an input stream and outputs the resulting packets
/// via output streams that can be read from using pollers. 
typedef struct mediapipe_instance mediapipe_instance;

/// Object for polling packets from an output stream.
typedef struct mediapipe_poller mediapipe_poller;

/// The basic data flow unit in a graph.
/// Packets can be put into an input stream to process them and polled from
/// an output stream to read the results.
typedef struct mediapipe_packet mediapipe_packet;

/// An image that can be processed by a graph.
typedef struct {
    const uint8_t* data;
    int width;
    int height;
    int format;
} mediapipe_image;

/// The 3D position of a landmark.
typedef struct {
    float x;
    float y;
    float z;
} mediapipe_landmark;

/// A list of landmarks detected for a face or hand.
typedef struct {
    mediapipe_landmark* elements;
    int length;
} mediapipe_landmark_list;

/// A list of hands or faces detcted in an image.
typedef struct {
    mediapipe_landmark_list* elements;
    int length;
} mediapipe_multi_face_landmark_list;

/// Hand landmark indices in a mediapipe_landmark_list.
/// For more information, see: https://google.github.io/mediapipe/solutions/hands.html#hand-landmark-model
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

/// Creates an instance builder, which is used by mediapipe_create_instance to create a MediaPipe instance.
/// The instance builder requires the path to the binary graph and the name of the input stream.
MEDIAPIPE_API mediapipe_instance_builder* mediapipe_create_instance_builder(const char* graph_filename, const char* input_stream);

/// Sets a float value in the node options of the graph.
MEDIAPIPE_API void mediapipe_add_option_float(mediapipe_instance_builder* instance_builder, const char* node, const char* option, float value);

/// Sets a double value in the node options of the graph.
MEDIAPIPE_API void mediapipe_add_option_double(mediapipe_instance_builder* instance_builder, const char* node, const char* option, double value);

/// Adds a side packet to the stream.
/// The function claims ownership of the packet and will deallocate it.
MEDIAPIPE_API void mediapipe_add_side_packet(mediapipe_instance_builder* instance_builder, const char* name, mediapipe_packet* packet);

/// Creates a MediaPipe instance, which represents the graph with some extra information.
/// The instance should be deallocated when no longer used with mediapipe_destroy_instance.
/// Returns NULL if the instance creation failed.
MEDIAPIPE_API mediapipe_instance* mediapipe_create_instance(mediapipe_instance_builder* builder);

/// Creates a poller to read packets from an output stream.
/// The poller should be deallocated before the instance with mediapipe_destroy_poller.
/// Returns NULL if the poller creation failed.
MEDIAPIPE_API mediapipe_poller* mediapipe_create_poller(mediapipe_instance* instance, const char* output_stream);

/// Starts the graph associated with the instance.
/// Returns true if starting the graph has succeeded.
MEDIAPIPE_API bool mediapipe_start(mediapipe_instance* instance);

/// Sends an image to the input stream specified in mediapipe_create_instance_builder.
/// The results won't be available immediately. Call mediapipe_get_queue_size to check whether the
/// image has been processed or mediapipe_wait_until_idle to block until the results are available. 
/// Returns true if submitting the image has succeeded.
MEDIAPIPE_API bool mediapipe_process(mediapipe_instance* instance, mediapipe_image image);

/// Wait until the MediaPipe graph has finished the work submitted with mediapipe_process and
/// the results are available.
/// Returns true if processing has succeeded.
MEDIAPIPE_API bool mediapipe_wait_until_idle(mediapipe_instance* instance);

/// Get the number of packets available in an output stream.
MEDIAPIPE_API int mediapipe_get_queue_size(mediapipe_poller* poller);

/// Deallocate a poller.
MEDIAPIPE_API void mediapipe_destroy_poller(mediapipe_poller* poller);

/// Deallocate an instance.
/// Returns true if deallocating the instance has succeeded.
MEDIAPIPE_API bool mediapipe_destroy_instance(mediapipe_instance* instance);

/// Set root resource directory where model files are loaded from. 
MEDIAPIPE_API void mediapipe_set_resource_dir(const char* dir);

/// Create a packet with an integer value.
/// The packet should be deallocated with mediapipe_destroy_packet unless used as a side packet.
MEDIAPIPE_API mediapipe_packet* mediapipe_create_packet_int(int value);

/// Create a packet with a float value.
/// The packet should be deallocated with mediapipe_destroy_packet unless used as a side packet.
MEDIAPIPE_API mediapipe_packet* mediapipe_create_packet_float(float value);

/// Create a packet with a bool value.
/// The packet should be deallocated with mediapipe_destroy_packet unless used as a side packet.
MEDIAPIPE_API mediapipe_packet* mediapipe_create_packet_bool(bool value);

/// Poll a packet from an output stream.
/// The packet should be deallocated with mediapipe_destroy_packet.
MEDIAPIPE_API mediapipe_packet* mediapipe_poll_packet(mediapipe_poller* poller);

/// Deallocates a packet. 
MEDIAPIPE_API void mediapipe_destroy_packet(mediapipe_packet* packet);

/// Gets the length of a packet type name without the NUL terminator.
MEDIAPIPE_API size_t mediapipe_get_packet_type_len(mediapipe_packet* packet);

/// Copies the name of a packet type into a buffer.
MEDIAPIPE_API void mediapipe_get_packet_type(mediapipe_packet* packet, char* buffer);

/// Copies an image of a packet into a buffer. 
MEDIAPIPE_API void mediapipe_copy_packet_image(mediapipe_packet* packet, uint8_t* out_data);

/// Returns the multi-face landmarks of a packet.
/// The multi-face landmark list should be destroyed with mediapipe_destroy_multi_face_landmarks.
MEDIAPIPE_API mediapipe_multi_face_landmark_list* mediapipe_get_multi_face_landmarks(mediapipe_packet* packet);

/// Returns the normalized multi-face landmarks of a packet.
/// The multi-face landmark list should be destroyed with mediapipe_destroy_multi_face_landmarks.
MEDIAPIPE_API mediapipe_multi_face_landmark_list* mediapipe_get_normalized_multi_face_landmarks(mediapipe_packet* packet);

/// Deallocates a multi-face landmark list
MEDIAPIPE_API void mediapipe_destroy_multi_face_landmarks(mediapipe_multi_face_landmark_list* multi_face_landmarks);

/// Prints the last error generated to stderr.
MEDIAPIPE_API void mediapipe_print_last_error();

#ifdef __cplusplus
}
#endif

#endif