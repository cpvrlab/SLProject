#ifndef APP
#define APP

#include "tools.h"

#define MERGE_SIMILAR_LOCATION 0

#define STOCK_ORBSLAM 0
#define STOCK_ORBSLAM_CLAHE 1
#define TILDE_BRIEF 2
#define TILDE_BRIEF_CLAHE 3
#define SURF_BRIEF 4
#define SURF_BRIEF_CLAHE 5
#define SURF_ORB 6
#define SURF_ORB_CLAHE 7

#define END_METHOD 8

#define USE_CLAHE 1


enum class InspectionMode
{
    MATCH_DRAWING_ALL = 49, //1
    MATCH_DRAWING_SINGLE,
    MATCHED_POINT_SIMILIARITY,
    ANY_KEYPOINT_COMPARISON,
    //ANY_PIXEL_COMPARISON,
    END
};

typedef struct App
{
    std::string name;
    //identifier for close up view
    std::string closeup_left;
    std::string closeup_right;
    //left image
    cv::Mat image1;
    //right image
    cv::Mat image2;
    //last selected keypoint index in the left image
    int left_idx = -1;
    //last selected keypoint index in the right image
    int right_idx = -1;

    PyramidParameters    pyramid_param;
    std::vector<cv::Mat> image1_pyramid;
    std::vector<cv::Mat> image2_pyramid;

    //extracted keypoints in left and right image
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    //extracted descriptors in left and right image corresponding to keypoint indices
    std::vector<Descriptor> descs1;
    std::vector<Descriptor> descs2;
    //look up best matching index in left image at index position of right image
    std::vector<int> matching_2_1;
    //look up best matching index in right image at index position of left image
    std::vector<int> matching_1_2;

    //key points in closest cartesian distance to last mouse click position
    std::vector<cv::KeyPoint> ordered_keypoints1;
    std::vector<cv::KeyPoint> ordered_keypoints2;
    //search radius for cartesian neighbour search after mouse click
    float select_radius = 10;
    //currently selected index for multi-click selection of nearest neighbour
    int local_idx = 0;

    //concatenated image of left and right image
    cv::Mat out_image;

    //currently selected mouse position
    cv::Point mouse_pos;
    //keyboard flags of last mouse click
    int keyboard_flags = 0;

    //feature extraction method
    int method = SURF_BRIEF;
    //current inspection mode (changable with keyboard numbers)
    InspectionMode inspectionMode = InspectionMode::MATCHED_POINT_SIMILIARITY;

    cv::Ptr<cv::CLAHE> clahe;

    //MATCHED_POINT_SIMILIARITY params:
    // number of best matches retrieved in MATCHED_POINT_SIMILIARITY
    int num_next_matches = 10;
    // stores if last click was left or right
    bool last_click_was_left = false;
    // currently selected next best match (index between 0 and num_next_matches-1)
    int curr_selected_match_idx = 0;
    // currently selected match in mouse wheel selection
    int next_sel_wheel = 0;
    // next matches description
    struct NextMatch
    {
        int        idx      = -1;
        float      distance = -1.f;
        cv::Scalar color;
        bool       operator<(const NextMatch& other)
        {
            return distance < other.distance;
        }
    };
    std::vector<NextMatch> next_matches;
} App;

void        app_next_method(App& app);
void        app_reset(App& app);
void        app_prepare(App& app);
std::string app_inspection_mode_text(App& app);

#endif
