#ifndef APP
#define APP

#include "tools.h"

#define EQUALIZE_HIST 0
#define MERGE_SIMILAR_LOCATION 0

#define STOCK_ORBSLAM 0
#define TILDE_BRIEF 1
#define SURF_BRIEF 2
#define SURF_ORB 3
#define END_METHOD 4

enum class InspectionMode
{
    MATCH_DRAWING_ALL = 49, //1
    MATCH_DRAWING_SINGLE,
    MATCHED_POINT_SIMILIARITY,
    ANY_KEYPOINT_COMPARISON,
    ANY_PIXEL_COMPARISON,
    END
};

typedef struct App
{
    std::string name;
    std::string closeup_left;
    std::string closeup_right;
    cv::Mat     image1;
    cv::Mat     image2;
    int         left_idx;
    int         right_idx;

    PyramidParameters    pyramid_param;
    std::vector<cv::Mat> image1_pyramid;
    std::vector<cv::Mat> image2_pyramid;

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<Descriptor>   descs1;
    std::vector<Descriptor>   descs2;
    std::vector<int>          matching_2_1;
    std::vector<int>          matching_1_2;

    std::vector<cv::Scalar> kp1_colors;
    std::vector<cv::Scalar> kp2_colors;

    std::vector<cv::KeyPoint> ordered_keypoints1;
    std::vector<cv::KeyPoint> ordered_keypoints2;

    cv::Mat out_image;

    cv::Point poi;
    int       local_idx;
    float     select_radius;

    //currently selected mouse position
    cv::Point mouse_pos;
    int       keyboard_flags = 0;

    int            method         = SURF_BRIEF;
    InspectionMode inspectionMode = InspectionMode::MATCH_DRAWING_ALL;

    std::string inspection_mode_text()
    {
        std::string text;
        switch (inspectionMode)
        {
            case InspectionMode::MATCH_DRAWING_ALL:
                return "All matches are visualized connected with lines";
            case InspectionMode::MATCH_DRAWING_SINGLE:
                return "Click on the image to visualize closed match by a single line. Close-up views for both keypoints are shown.";
            case InspectionMode::MATCHED_POINT_SIMILIARITY:
                return "Click on left or right image to catch the closest feature point. Closest feature points in the other image are highlighted\nand a close-up of the catched keypoint is drawn. (Multi-click on the same position iterates keypoints in the neighbourhood.)";
            case InspectionMode::ANY_PIXEL_COMPARISON:
                return "Click on both images to select two pixel positons. The descriptors at selected positions are compared and visualized in close-up views";
            case InspectionMode::ANY_KEYPOINT_COMPARISON:
                return "Click on both images to catch two keypoints. The descriptors at selected positions are compared and visualized in close-up views. (Multi-click on the same position iterates keypoints in the neighbourhood.)";
            default:
                return "";
        }
    }
} App;

void app_next_method(App& app);
void app_reset(App& app);
void app_prepare(App& app);

#endif
