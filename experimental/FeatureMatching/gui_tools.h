#ifndef GUI_TOOLS
#define GUI_TOOLS
#include "tools.h"
#include "app.h"

//Color in BGR
cv::Scalar red();
cv::Scalar blue();
cv::Scalar green();
cv::Scalar white();

cv::Scalar color_interpolate(cv::Scalar c1, cv::Scalar c2, float v);

void reset_similarity(std::vector<cv::KeyPoint>& kps);

void reset_color(std::vector<cv::Scalar>& colors, cv::Scalar col);

void init_color(std::vector<cv::Scalar>& colors, int size);

void set_color_by_value(std::vector<cv::Scalar>& colors, std::vector<cv::KeyPoint>& kps);

void draw_matches_lines(App& app, const cv::Scalar color);

void draw_concat_images(App& app);

void draw_all_keypoins(App& app, cv::Scalar color);

void draw_matched_keypoints(App& app, cv::Scalar color);

void draw_selected_keypoints(App& app);

void draw_match_line(App& app, int matchIndex1, int matchIndex2, cv::Scalar color);

void draw_by_similarity(App& app);

void draw_closeup_right(App& app, int idx);

void draw_closeup_left(App& app, int idx);

void draw_main(App& app);

#endif
