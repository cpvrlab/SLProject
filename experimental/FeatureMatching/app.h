#ifndef APP
#define APP

#include "tools.h"

#define EQUALIZE_HIST   0
#define MERGE_SIMILAR_LOCATION  0

#define STOCK_ORBSLAM   0
#define TILDE_BRIEF     1
#define SURF_BRIEF      2
#define SURF_ORB        3
#define END_METHOD      4

typedef struct App
{
    std::string name;
    std::string closeup_left;
    std::string closeup_right;
    cv::Mat image1;
    cv::Mat image2;
    int left_idx;
    int right_idx;

    PyramidParameters pyramid_param;
    std::vector<cv::Mat> image1_pyramid;
    std::vector<cv::Mat> image2_pyramid;

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<Descriptor> descs1;
    std::vector<Descriptor> descs2;
    std::vector<int> matching_2_1;
    std::vector<int> matching_1_2;

    std::vector<cv::Scalar> kp1_colors;
    std::vector<cv::Scalar> kp2_colors;

    std::vector<cv::KeyPoint> ordered_keypoints1;
    std::vector<cv::KeyPoint> ordered_keypoints2;

    cv::Mat out_image;

    cv::Point poi;
    int local_idx;
    float select_radius;

    int method;
}App;


void app_next_method(App &app);

void app_reset(App &app);

void app_prepare(App &app);

#endif

