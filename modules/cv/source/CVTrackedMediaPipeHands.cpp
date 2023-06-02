//#############################################################################
//  File:      CVTrackedMediaPipeHands.cpp
//  Date:      December 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_BUILD_WITH_MEDIAPIPE
#    include <CVTrackedMediaPipeHands.h>

//-----------------------------------------------------------------------------
#    define CHECK_MP_RESULT(result) \
        if (!result) \
        { \
            const char* error = mp_get_last_error(); \
            std::cerr << error << std::endl; \
            mp_free_error(error); \
            SL_EXIT_MSG("Exiting due to MediaPipe error"); \
        }
//-----------------------------------------------------------------------------
typedef std::vector<std::pair<mp_hand_landmark, mp_hand_landmark>> ConnectionList;
//-----------------------------------------------------------------------------
//! ???
/*!
 * ??? With MediaPipe docs links
 * Defines the connection list used for drawing the hand skeleton
 */
static const ConnectionList CONNECTIONS = {
  {mp_hand_landmark_wrist, mp_hand_landmark_thumb_cmc},
  {mp_hand_landmark_thumb_cmc, mp_hand_landmark_thumb_mcp},
  {mp_hand_landmark_thumb_mcp, mp_hand_landmark_thumb_ip},
  {mp_hand_landmark_thumb_ip, mp_hand_landmark_thumb_tip},
  {mp_hand_landmark_wrist, mp_hand_landmark_index_finger_mcp},
  {mp_hand_landmark_index_finger_mcp, mp_hand_landmark_index_finger_pip},
  {mp_hand_landmark_index_finger_pip, mp_hand_landmark_index_finger_dip},
  {mp_hand_landmark_index_finger_dip, mp_hand_landmark_index_finger_tip},
  {mp_hand_landmark_index_finger_mcp, mp_hand_landmark_middle_finger_mcp},
  {mp_hand_landmark_middle_finger_mcp, mp_hand_landmark_middle_finger_pip},
  {mp_hand_landmark_middle_finger_pip, mp_hand_landmark_middle_finger_dip},
  {mp_hand_landmark_middle_finger_dip, mp_hand_landmark_middle_finger_tip},
  {mp_hand_landmark_middle_finger_mcp, mp_hand_landmark_ring_finger_mcp},
  {mp_hand_landmark_ring_finger_mcp, mp_hand_landmark_ring_finger_pip},
  {mp_hand_landmark_ring_finger_pip, mp_hand_landmark_ring_finger_dip},
  {mp_hand_landmark_ring_finger_dip, mp_hand_landmark_ring_finger_tip},
  {mp_hand_landmark_ring_finger_mcp, mp_hand_landmark_pinky_mcp},
  {mp_hand_landmark_wrist, mp_hand_landmark_pinky_mcp},
  {mp_hand_landmark_pinky_mcp, mp_hand_landmark_pinky_pip},
  {mp_hand_landmark_pinky_pip, mp_hand_landmark_pinky_dip},
  {mp_hand_landmark_pinky_dip, mp_hand_landmark_pinky_tip}};
//-----------------------------------------------------------------------------
CVTrackedMediaPipeHands::CVTrackedMediaPipeHands(SLstring dataPath)
{
    mp_set_resource_dir(dataPath.c_str());

    SLstring graphPath = dataPath +
                         "mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb";
    auto* builder = mp_create_instance_builder(
      graphPath.c_str(),
      "image");

    // ??? What is the effect of this parameter
    mp_add_option_float(builder,
                        "palmdetectioncpu__TensorsToDetectionsCalculator",
                        "min_score_thresh",
                        0.5);

    // ??? What is the effect of this parameter
    mp_add_option_double(builder,
                         "handlandmarkcpu__ThresholdingCalculator",
                         "threshold",
                         0.5);

    // ??? What is the effect of this parameter
    mp_add_side_packet(builder,
                       "num_hands",
                       mp_create_packet_int(2));

    // ??? What is the effect of this parameter
    mp_add_side_packet(builder,
                       "model_complexity",
                       mp_create_packet_int(1));

    // ??? What is the effect of this parameter
    mp_add_side_packet(builder,
                       "use_prev_landmarks",
                       mp_create_packet_bool(true));

    // Creates a MediaPipe instance with the graph and some extra info
    _instance = mp_create_instance(builder);
    CHECK_MP_RESULT(_instance)

    // Creates a poller to read packets from an output stream.
    _landmarksPoller = mp_create_poller(_instance,
                                        "multi_hand_landmarks");
    CHECK_MP_RESULT(_landmarksPoller)

    // Starts the MediaPipe graph
    CHECK_MP_RESULT(mp_start(_instance))

    // clang-format off
    // We define a identity matrix for the object view matrix because we do
    // not transform any object in the scenegraph so far.
    _objectViewMat = CVMatx44f(1,0,0,0,
                               0,1,0,0,
                               0,0,1,0,
                               0,0,0,1);
    // clang-format on
}
//-----------------------------------------------------------------------------
CVTrackedMediaPipeHands::~CVTrackedMediaPipeHands()
{
    mp_destroy_poller(_landmarksPoller);
    CHECK_MP_RESULT(mp_destroy_instance(_instance))
}
//-----------------------------------------------------------------------------
bool CVTrackedMediaPipeHands::track(CVMat          imageGray,
                                    CVMat          imageRgb,
                                    CVCalibration* calib)
{
    processImageInMediaPipe(imageRgb);

    if (mp_get_queue_size(_landmarksPoller) > 0)
    {
        auto* landmarksPacket = mp_poll_packet(_landmarksPoller);
        auto* landmarks       = mp_get_norm_multi_face_landmarks(landmarksPacket);

        drawResults(landmarks, imageRgb);

        mp_destroy_multi_face_landmarks(landmarks);
        mp_destroy_packet(landmarksPacket);
    }

    return true;
}
//-----------------------------------------------------------------------------
void CVTrackedMediaPipeHands::processImageInMediaPipe(CVMat imageRgb)
{
    mp_image in_image;
    in_image.data     = imageRgb.data;
    in_image.width    = imageRgb.cols;
    in_image.height   = imageRgb.rows;
    in_image.format   = mp_image_format_srgb;
    mp_packet* packet = mp_create_packet_image(in_image);


    CHECK_MP_RESULT(mp_process(_instance, packet))

    // ???
    CHECK_MP_RESULT(mp_wait_until_idle(_instance))
}
//-----------------------------------------------------------------------------
//! Draws the hand skeleton with connections and joints into the RGB image
void CVTrackedMediaPipeHands::drawResults(mp_multi_face_landmark_list* landmarks,
                                          CVMat                        imageRgb)
{
    for (int i = 0; i < landmarks->length; i++)
    {
        auto& hand = landmarks->elements[i];

        for (auto& connection : CONNECTIONS)
        {
            auto  p1 = hand.elements[connection.first];
            auto  p2 = hand.elements[connection.second];
            float x1 = (float)imageRgb.cols * p1.x;
            float y1 = (float)imageRgb.rows * p1.y;
            float x2 = (float)imageRgb.cols * p2.x;
            float y2 = (float)imageRgb.rows * p2.y;

            cv::line(imageRgb,
                     {(int)x1, (int)y1},
                     {(int)x2, (int)y2},
                     CV_RGB(0, 255, 0),
                     2);
        }

        for (int j = 0; j < hand.length; j++)
        {
            auto  p      = hand.elements[j];
            float x      = (float)imageRgb.cols * p.x;
            float y      = (float)imageRgb.rows * p.y;
            float radius = std::max(3.0f + 25.0f * -p.z, 1.0f);

            cv::circle(imageRgb,
                       CVPoint((int)x, (int)y),
                       (int)radius,
                       CV_RGB(255, 0, 0),
                       -1);
        }
    }
}
//-----------------------------------------------------------------------------
#endif