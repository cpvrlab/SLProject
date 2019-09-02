#include "gui_tools.h"
#include "app.h"


cv::Scalar red() { return cv::Scalar(0, 0, 255); }
cv::Scalar blue() { return cv::Scalar(255, 0, 0); }
cv::Scalar green() { return cv::Scalar(0, 255, 0); }
cv::Scalar white() { return cv::Scalar(255, 255, 255); }

cv::Scalar color_interpolate(cv::Scalar c1, cv::Scalar c2, float v)
{
    return (1-v) * c1 + v * c2;
}

void reset_similarity(std::vector<cv::KeyPoint> &kps)
{
    for (cv::KeyPoint &kp : kps)
    {
        kp.response = 1.0;
    }
}

void reset_color(std::vector<cv::Scalar> &colors, cv::Scalar col)
{
    for (int i = 0; i < colors.size(); i++)
    {
        colors[i] = col;
    }
}

void init_color(std::vector<cv::Scalar> &colors, int size)
{
    colors.resize(size);
    reset_color(colors, blue());
}

void set_color_by_value(std::vector<cv::Scalar> &colors, std::vector<cv::KeyPoint> &kps)
{
    for (int i = 0; i < kps.size(); i++)
    {
        colors[i] = color_interpolate(blue(), red(), kps[i].response);
    }
}

void draw_matches_lines(App &app)
{
    cv::hconcat(app.image1, app.image2, app.out_image);

    for (int i = 0; i < app.keypoints1.size(); i++)
    {
        cv::circle(app.out_image, app.keypoints1[i].pt, 3, app.kp1_colors[i], 1);
    }

    for (int i = 0; i < app.keypoints2.size(); i++)
    {
        cv::circle(app.out_image, app.keypoints2[i].pt + cv::Point2f(app.image1.cols, 0), 3, app.kp2_colors[i], 1);
    }

    for (int i = 0; i < app.matching_2_1.size(); i++)
    {
        if (app.matching_2_1[i] >= 0)
        {
            cv::line(app.out_image, app.keypoints2[i].pt + cv::Point2f(app.image1.cols, 0), app.keypoints1[app.matching_2_1[i]].pt, app.kp2_colors[i], 2);
        }
    }
}

void draw_by_similarity(App &app)
{
    cv::hconcat(app.image1, app.image2, app.out_image);

    for (int i = 0; i < app.ordered_keypoints1.size(); i++)
    {
        cv::circle(app.out_image, app.ordered_keypoints1[i].pt, 3, app.kp1_colors[i], 1);
        if (app.ordered_keypoints1[i].response < 0.5)
        {
            cv::circle(app.out_image, app.ordered_keypoints1[i].pt, 15, app.kp1_colors[i], 3);
        }
    }
    for (int i = 0; i < app.keypoints2.size(); i++)
    {
        cv::circle(app.out_image, app.ordered_keypoints2[i].pt + cv::Point2f(app.image1.cols, 0), 3, app.kp2_colors[i], 1);
        if (app.ordered_keypoints2[i].response < 0.5)
        {
            cv::circle(app.out_image, app.ordered_keypoints2[i].pt + cv::Point2f(app.image1.cols, 0), 15, app.kp2_colors[i], 3);
        }
    }
}


