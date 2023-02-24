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

/// Structure specifying instance properties before creation.
typedef struct mp_instance_builder mp_instance_builder;

/// Structure containing the MediaPipe graph as well as some extra information.
/// Data in MediaPipe flows between nodes of a graph to perform computations.
/// The graph takes some packets via an input stream and outputs the resulting packets
/// via output streams that can be read from using pollers.
typedef struct mp_instance mp_instance;

/// An interface for polling packets from an output stream.
typedef struct mp_poller mp_poller;

/// The basic data flow unit in a graph.
/// Packets can be put into an input stream to process them and polled from
/// an output stream to access the results.
typedef struct mp_packet mp_packet;

/// Enum specifying the format of image data.
/// For more information, see: https://github.com/google/mediapipe/blob/master/mediapipe/framework/formats/image_format.proto
typedef enum : int {
  mp_image_format_unknown = 0,
  mp_image_format_srgb = 1,
  mp_image_format_srgba = 2,
  mp_image_format_gray8 = 3,
  mp_image_format_gray16 = 4,
  mp_image_format_ycbcr420p = 5,
  mp_image_format_ycbcr420p10 = 6,
  mp_image_format_srgb48 = 7,
  mp_image_format_srgba64 = 8,
  mp_image_format_vec32f1 = 9,
  mp_image_format_vec32f2 = 12,
  mp_image_format_lab8 = 10,
  mp_image_format_sbgra = 11
} mp_image_format;

/// A structure for wrapping pixel data.
typedef struct {
    const uint8_t* data;
    int width;
    int height;
    mp_image_format format;
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

/// A list of hands or faces detected in an image.
typedef struct {
    mp_landmark_list* elements;
    int length;
} mp_multi_face_landmark_list;

/// A rectangle with a rotation in radians.
typedef struct {
    float x_center;
    float y_center;
    float width;
    float height;
    float rotation;
    long long id;
} mp_rect;

/// A list of rectangles.
typedef struct {
    mp_rect* elements;
    int length;
} mp_rect_list;

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

/// Creates a MediaPipe instance, which contains the graph with some extra information.
/// The instance should be deallocated when no longer used with mp_destroy_instance.
/// Returns NULL on failure.
MEDIAPIPE_API mp_instance* mp_create_instance(mp_instance_builder* builder);

/// Creates a poller to read packets from an output stream.
/// The poller should be deallocated before the instance with mp_destroy_poller.
/// Returns NULL on failure.
MEDIAPIPE_API mp_poller* mp_create_poller(mp_instance* instance, const char* output_stream);

/// Starts the graph associated with the instance.
/// Returns true on success.
MEDIAPIPE_API bool mp_start(mp_instance* instance);

/// Sends a packet to the input stream specified in mp_create_instance_builder.
/// The results won't be available immediately. Call mp_get_queue_size to check whether the
/// packet has been processed or mp_wait_until_idle to block until the results are available.
/// Returns true on success.
MEDIAPIPE_API bool mp_process(mp_instance* instance, mp_packet* packet);

/// Blocks until the MediaPipe graph has finished the work submitted with mp_process and
/// the results are available.
/// Returns true on success.
MEDIAPIPE_API bool mp_wait_until_idle(mp_instance* instance);

/// Returns the number of packets available in an output stream.
/// This function should be called before polling packets to avoid infinite blocking.
MEDIAPIPE_API int mp_get_queue_size(mp_poller* poller);

/// Deallocates a poller.
MEDIAPIPE_API void mp_destroy_poller(mp_poller* poller);

/// Stops the graph associated with the instance and deallocates it.
/// Returns true on success.
MEDIAPIPE_API bool mp_destroy_instance(mp_instance* instance);

/// Sets the root resource directory.
/// Model files referenced in graphs are loaded from this directory.
MEDIAPIPE_API void mp_set_resource_dir(const char* dir);

/// Creates a packet with an integer value.
MEDIAPIPE_API mp_packet* mp_create_packet_int(int value);

/// Creates a packet with a float value.
MEDIAPIPE_API mp_packet* mp_create_packet_float(float value);

/// Creates a packet with a bool value.
MEDIAPIPE_API mp_packet* mp_create_packet_bool(bool value);

/// Creates a packet with an image value.
/// The pixel data is copied and not deallocated.
MEDIAPIPE_API mp_packet* mp_create_packet_image(mp_image image);

/// Polls a packet from an output stream.
/// The packet should be deallocated with mp_destroy_packet.
MEDIAPIPE_API mp_packet* mp_poll_packet(mp_poller* poller);

/// Deallocates a packet.
MEDIAPIPE_API void mp_destroy_packet(mp_packet* packet);

/// Returns the name of the packet type.
MEDIAPIPE_API const char* mp_get_packet_type(mp_packet* packet);

/// Deallocates the type name returned by mp_get_packet_type.
MEDIAPIPE_API void mp_free_packet_type(const char* type);

/// Copies a packet image into a buffer.
MEDIAPIPE_API void mp_copy_packet_image(mp_packet* packet, uint8_t* out_data);

/// Returns the multi-face landmarks of a packet.
/// The list should be destroyed with mp_destroy_multi_face_landmarks.
MEDIAPIPE_API mp_multi_face_landmark_list* mp_get_multi_face_landmarks(mp_packet* packet);

/// Returns the normalized multi-face landmarks of a packet.
/// The list should be destroyed with mp_destroy_multi_face_landmarks.
MEDIAPIPE_API mp_multi_face_landmark_list* mp_get_norm_multi_face_landmarks(mp_packet* packet);

/// Deallocates a multi-face landmark list.
MEDIAPIPE_API void mp_destroy_multi_face_landmarks(mp_multi_face_landmark_list* multi_face_landmarks);

/// Returns the rectangles of a packet.
/// The list should be destroyed with mp_destroy_rects.
MEDIAPIPE_API mp_rect_list* mp_get_rects(mp_packet* packet);

/// Returns the normalized rectangles of a packet.
/// The list should be destroyed with mp_destroy_rects.
MEDIAPIPE_API mp_rect_list* mp_get_norm_rects(mp_packet* packet);

/// Deallocates a rectangle list.
MEDIAPIPE_API void mp_destroy_rects(mp_rect_list* list);

/// Returns the last error message generated.
MEDIAPIPE_API const char* mp_get_last_error();

/// Deallocates the error message returned by mp_get_last_error
MEDIAPIPE_API void mp_free_error(const char* message);

#ifdef __cplusplus
}
#endif

#endif