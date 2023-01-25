//#############################################################################
//  File:      CVTrackedMediaPipeHands.cpp
//  Date:      December 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "CVTrackedMediaPipeHands.h"

//-----------------------------------------------------------------------------
#define CHECK_MP_RESULT(result) \
    if (!result) \
    { \
        mediapipe_print_last_error(); \
        SL_EXIT_MSG("Exiting due to MediaPipe error"); \
    }
//-----------------------------------------------------------------------------
typedef std::vector<std::pair<mediapipe_hand_landmark, mediapipe_hand_landmark>> ConnectionList;
//-----------------------------------------------------------------------------
static const ConnectionList CONNECTIONS = {{mediapipe_hand_landmark_wrist, mediapipe_hand_landmark_thumb_cmc},
                                           {mediapipe_hand_landmark_thumb_cmc, mediapipe_hand_landmark_thumb_mcp},
                                           {mediapipe_hand_landmark_thumb_mcp, mediapipe_hand_landmark_thumb_ip},
                                           {mediapipe_hand_landmark_thumb_ip, mediapipe_hand_landmark_thumb_tip},
                                           {mediapipe_hand_landmark_wrist, mediapipe_hand_landmark_index_finger_mcp},
                                           {mediapipe_hand_landmark_index_finger_mcp, mediapipe_hand_landmark_index_finger_pip},
                                           {mediapipe_hand_landmark_index_finger_pip, mediapipe_hand_landmark_index_finger_dip},
                                           {mediapipe_hand_landmark_index_finger_dip, mediapipe_hand_landmark_index_finger_tip},
                                           {mediapipe_hand_landmark_index_finger_mcp, mediapipe_hand_landmark_middle_finger_mcp},
                                           {mediapipe_hand_landmark_middle_finger_mcp, mediapipe_hand_landmark_middle_finger_pip},
                                           {mediapipe_hand_landmark_middle_finger_pip, mediapipe_hand_landmark_middle_finger_dip},
                                           {mediapipe_hand_landmark_middle_finger_dip, mediapipe_hand_landmark_middle_finger_tip},
                                           {mediapipe_hand_landmark_middle_finger_mcp, mediapipe_hand_landmark_ring_finger_mcp},
                                           {mediapipe_hand_landmark_ring_finger_mcp, mediapipe_hand_landmark_ring_finger_pip},
                                           {mediapipe_hand_landmark_ring_finger_pip, mediapipe_hand_landmark_ring_finger_dip},
                                           {mediapipe_hand_landmark_ring_finger_dip, mediapipe_hand_landmark_ring_finger_tip},
                                           {mediapipe_hand_landmark_ring_finger_mcp, mediapipe_hand_landmark_pinky_mcp},
                                           {mediapipe_hand_landmark_wrist, mediapipe_hand_landmark_pinky_mcp},
                                           {mediapipe_hand_landmark_pinky_mcp, mediapipe_hand_landmark_pinky_pip},
                                           {mediapipe_hand_landmark_pinky_pip, mediapipe_hand_landmark_pinky_dip},
                                           {mediapipe_hand_landmark_pinky_dip, mediapipe_hand_landmark_pinky_tip}};
//-----------------------------------------------------------------------------
CVTrackedMediaPipeHands::CVTrackedMediaPipeHands(SLstring dataPath)
{
    mediapipe_set_resource_dir(dataPath.c_str());

    SLstring graphPath = dataPath + "mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb";
    auto* builder = mediapipe_create_instance_builder(graphPath.c_str(), "image");
    mediapipe_add_option_float(builder, "palmdetectioncpu__TensorsToDetectionsCalculator", "min_score_thresh", 0.5);
    mediapipe_add_option_double(builder, "handlandmarkcpu__ThresholdingCalculator", "threshold", 0.5);
    mediapipe_add_side_packet(builder, "num_hands", mediapipe_create_packet_int(2));
    mediapipe_add_side_packet(builder, "model_complexity", mediapipe_create_packet_int(1));
    mediapipe_add_side_packet(builder, "use_prev_landmarks", mediapipe_create_packet_bool(true));

    _instance = mediapipe_create_instance(builder);
    CHECK_MP_RESULT(_instance)

    _landmarksPoller = mediapipe_create_poller(_instance, "multi_hand_landmarks");
    CHECK_MP_RESULT(_landmarksPoller)

    CHECK_MP_RESULT(mediapipe_start(_instance))
}
//-----------------------------------------------------------------------------
CVTrackedMediaPipeHands::~CVTrackedMediaPipeHands()
{
    CHECK_MP_RESULT(mediapipe_destroy_instance(_instance))
}
//-----------------------------------------------------------------------------
bool CVTrackedMediaPipeHands::track(CVMat          imageGray,
                                    CVMat          imageRgb,
                                    CVCalibration* calib)
{
    processImage(imageRgb);

    if (mediapipe_get_queue_size(_landmarksPoller) > 0)
    {
        auto* landmarksPacket = mediapipe_poll_packet(_landmarksPoller);
        auto* landmarks       = mediapipe_get_normalized_multi_face_landmarks(landmarksPacket);

        drawResults(landmarks, imageRgb);

        mediapipe_destroy_multi_face_landmarks(landmarks);
        mediapipe_destroy_packet(landmarksPacket);
    }

    _objectViewMat = CVMatx44f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    return true;
}
//-----------------------------------------------------------------------------
void CVTrackedMediaPipeHands::processImage(CVMat imageRgb)
{
    mediapipe_image in_image;
    in_image.data   = imageRgb.data;
    in_image.width  = imageRgb.cols;
    in_image.height = imageRgb.rows;
    in_image.format = 1;

    CHECK_MP_RESULT(mediapipe_process(_instance, in_image))
    CHECK_MP_RESULT(mediapipe_wait_until_idle(_instance))
}
//-----------------------------------------------------------------------------
void CVTrackedMediaPipeHands::drawResults(mediapipe_multi_face_landmark_list* landmarks,
                                          CVMat                               imageRgb)
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

            cv::line(imageRgb, {(int)x1, (int)y1}, {(int)x2, (int)y2}, CV_RGB(0, 255, 0), 2);
        }

        for (int j = 0; j < hand.length; j++)
        {
            auto  p      = hand.elements[j];
            float x      = (float)imageRgb.cols * p.x;
            float y      = (float)imageRgb.rows * p.y;
            float radius = 3.0f + 25.0f * -p.z;

            cv::circle(imageRgb, CVPoint((int)x, (int)y), (int)radius, CV_RGB(255, 0, 0), -1);
        }
    }
}
//-----------------------------------------------------------------------------