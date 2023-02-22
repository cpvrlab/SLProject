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
typedef struct mp_instance_builder mp_instance_builder;

/// Contains the MediaPipe graph as well as some extra information.
/// Data in MediaPipe flows between node of a graph to perform computations.
/// The graph takes some packets via an input stream and outputs the resulting packets
/// via output streams that can be read from using pollers. 
typedef struct mp_instance mp_instance;

/// Object for polling packets from an output stream.
typedef struct mp_poller mp_poller;

/// The basic data flow unit in a graph.
/// Packets can be put into an input stream to process them and polled from
/// an output stream to read the results.
typedef struct mp_packet mp_packet;

/// An image that can be processed by a graph.
typedef struct {
    const uint8_t* data;
    int width;
    int height;
    int format;
} mp_image;

/// The 3D position of a landmark.
typedef struct {
    float x;
    float y;
    float z;
} mp_landmark;

/// A list of landmarks detected for a face or hand.
typedef struct {
    mp_landmark* elements;
    int length;
} mp_landmark_list;

/// A list of hands or faces detcted in an image.
typedef struct {
    mp_landmark_list* elements;
    int length;
} mp_multi_face_landmark_list;

/// Hand landmark indices in a mp_landmark_list.
/// For more information, see: https://google.github.io/mediapipe/solutions/hands.html#hand-landmark-model
typedef enum {
    mp_hand_landmark_wrist = 0,
    mp_hand_landmark_thumb_cmc = 1,
    mp_hand_landmark_thumb_mcp = 2,
    mp_hand_landmark_thumb_ip = 3,
    mp_hand_landmark_thumb_tip = 4,
    mp_hand_landmark_index_finger_mcp = 5,
    mp_hand_landmark_index_finger_pip = 6,
    mp_hand_landmark_index_finger_dip = 7,
    mp_hand_landmark_index_finger_tip = 8,
    mp_hand_landmark_middle_finger_mcp = 9,
    mp_hand_landmark_middle_finger_pip = 10,
    mp_hand_landmark_middle_finger_dip = 11,
    mp_hand_landmark_middle_finger_tip = 12,
    mp_hand_landmark_ring_finger_mcp = 13,
    mp_hand_landmark_ring_finger_pip = 14,
    mp_hand_landmark_ring_finger_dip = 15,
    mp_hand_landmark_ring_finger_tip = 16,
    mp_hand_landmark_pinky_mcp = 17,
    mp_hand_landmark_pinky_pip = 18,
    mp_hand_landmark_pinky_dip = 19,
    mp_hand_landmark_pinky_tip = 20
} mp_hand_landmark;

/// Creates an instance builder, which is used by mp_create_instance to create a MediaPipe instance.
/// The instance builder requires the path to the binary graph and the name of the input stream.
MEDIAPIPE_API mp_instance_builder* mp_create_instance_builder(const char* graph_filename, const char* input_stream);

/// Sets a float value in the node options of the graph.
MEDIAPIPE_API void mp_add_option_float(mp_instance_builder* instance_builder, const char* node, const char* option, float value);

/// Sets a double value in the node options of the graph.
MEDIAPIPE_API void mp_add_option_double(mp_instance_builder* instance_builder, const char* node, const char* option, double value);

/// Adds a side packet to the stream.
/// The function claims ownership of the packet and will deallocate it.
MEDIAPIPE_API void mp_add_side_packet(mp_instance_builder* instance_builder, const char* name, mp_packet* packet);

/// Creates a MediaPipe instance, which represents the graph with some extra information.
/// The instance should be deallocated when no longer used with mp_destroy_instance.
/// Returns NULL if the instance creation failed.
MEDIAPIPE_API mp_instance* mp_create_instance(mp_instance_builder* builder);

/// Creates a poller to read packets from an output stream.
/// The poller should be deallocated before the instance with mp_destroy_poller.
/// Returns NULL if the poller creation failed.
MEDIAPIPE_API mp_poller* mp_create_poller(mp_instance* instance, const char* output_stream);

/// Starts the graph associated with the instance.
/// Returns true if starting the graph has succeeded.
MEDIAPIPE_API bool mp_start(mp_instance* instance);

/// Sends a packet to the input stream specified in mp_create_instance_builder.
/// The results won't be available immediately. Call mp_get_queue_size to check whether the
/// packet has been processed or mp_wait_until_idle to block until the results are available. 
/// Returns true if submitting the packet has succeeded.
MEDIAPIPE_API bool mp_process(mp_instance* instance, mp_packet* packet);

/// Wait until the MediaPipe graph has finished the work submitted with mp_process and
/// the results are available.
/// Returns true if processing has succeeded.
MEDIAPIPE_API bool mp_wait_until_idle(mp_instance* instance);

/// Get the number of packets available in an output stream.
MEDIAPIPE_API int mp_get_queue_size(mp_poller* poller);

/// Deallocate a poller.
MEDIAPIPE_API void mp_destroy_poller(mp_poller* poller);

/// Deallocate an instance.
/// Returns true if deallocating the instance has succeeded.
MEDIAPIPE_API bool mp_destroy_instance(mp_instance* instance);

/// Set root resource directory where model files are loaded from. 
MEDIAPIPE_API void mp_set_resource_dir(const char* dir);

/// Create a packet with an integer value.
/// The packet should be deallocated with mp_destroy_packet if polled from an output stream.
MEDIAPIPE_API mp_packet* mp_create_packet_int(int value);

/// Create a packet with a float value.
/// The packet should be deallocated with mp_destroy_packet if polled from an output stream.
MEDIAPIPE_API mp_packet* mp_create_packet_float(float value);

/// Create a packet with a bool value.
/// The packet should be deallocated with mp_destroy_packet if polled from an output stream.
MEDIAPIPE_API mp_packet* mp_create_packet_bool(bool value);

/// Create a packet with an image value.
/// The packet should be deallocated with mp_destroy_packet if polled from an output stream.
/// The image data is copied and should be deallocated by the caller.
MEDIAPIPE_API mp_packet* mp_create_packet_image(mp_image image);

/// Poll a packet from an output stream.
/// The packet should be deallocated with mp_destroy_packet.
MEDIAPIPE_API mp_packet* mp_poll_packet(mp_poller* poller);

/// Deallocates a packet. 
MEDIAPIPE_API void mp_destroy_packet(mp_packet* packet);

/// Gets the length of a packet type name without the NUL terminator.
MEDIAPIPE_API size_t mp_get_packet_type_len(mp_packet* packet);

/// Copies the name of a packet type into a buffer.
MEDIAPIPE_API void mp_get_packet_type(mp_packet* packet, char* buffer);

/// Copies an image of a packet into a buffer. 
MEDIAPIPE_API void mp_copy_packet_image(mp_packet* packet, uint8_t* out_data);

/// Returns the multi-face landmarks of a packet.
/// The multi-face landmark list should be destroyed with mp_destroy_multi_face_landmarks.
MEDIAPIPE_API mp_multi_face_landmark_list* mp_get_multi_face_landmarks(mp_packet* packet);

/// Returns the normalized multi-face landmarks of a packet.
/// The multi-face landmark list should be destroyed with mp_destroy_multi_face_landmarks.
MEDIAPIPE_API mp_multi_face_landmark_list* mp_get_normalized_multi_face_landmarks(mp_packet* packet);

/// Deallocates a multi-face landmark list
MEDIAPIPE_API void mp_destroy_multi_face_landmarks(mp_multi_face_landmark_list* multi_face_landmarks);

/// Prints the last error generated to stderr.
MEDIAPIPE_API void mp_print_last_error();

#ifdef __cplusplus
}
#endif

#endif