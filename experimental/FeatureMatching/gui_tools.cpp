#include "gui_tools.h"
#include "app.h"

cv::Scalar red() { return cv::Scalar(0, 0, 255); }
cv::Scalar blue() { return cv::Scalar(255, 0, 0); }
cv::Scalar green() { return cv::Scalar(0, 255, 0); }
cv::Scalar white() { return cv::Scalar(255, 255, 255); }

cv::Scalar color_interpolate(cv::Scalar c1, cv::Scalar c2, float v)
{
    return (1 - v) * c1 + v * c2;
}

//void reset_similarity(std::vector<cv::KeyPoint>& kps)
//{
//    for (cv::KeyPoint& kp : kps)
//    {
//        kp.response = 1.0;
//    }
//}

void reset_color(std::vector<cv::Scalar>& colors, cv::Scalar col)
{
    for (int i = 0; i < colors.size(); i++)
    {
        colors[i] = col;
    }
}

void init_color(std::vector<cv::Scalar>& colors, int size)
{
    colors.resize(size);
    reset_color(colors, blue());
}

//void set_color_by_value(std::vector<cv::Scalar>& colors, std::vector<cv::KeyPoint>& kps)
//{
//    for (int i = 0; i < kps.size(); i++)
//    {
//        colors[i] = color_interpolate(blue(), red(), kps[i].response);
//    }
//}

//draw concatenated image
void draw_concat_images(App& app)
{
    cv::hconcat(app.image1, app.image2, app.out_image);
}

void draw_all_keypoins(App& app, cv::Scalar color)
{
    for (int i = 0; i < app.keypoints1.size(); i++)
    {
        cv::circle(app.out_image, app.keypoints1[i].pt, 3, color, 1);
    }

    for (int i = 0; i < app.keypoints2.size(); i++)
    {
        cv::circle(app.out_image, app.keypoints2[i].pt + cv::Point2f(app.image1.cols, 0), 3, color, 1);
    }
}

void draw_matched_keypoints(App& app, cv::Scalar color)
{
    cv::Point2f offset(app.image1.cols, 0);
    for (int i = 0; i < app.matching_2_1.size(); i++)
    {
        if (app.matching_2_1[i] >= 0)
        {
            cv::circle(app.out_image, app.keypoints2[i].pt + offset, 3, color, 1);
            cv::circle(app.out_image, app.keypoints1[app.matching_2_1[i]].pt, 3, color, 1);
        }
    }
}

void draw_selected_keypoints(App& app)
{
    if (app.left_idx >= 0 && app.left_idx < app.keypoints1.size())
    {
        cv::circle(app.out_image, app.keypoints1[app.left_idx].pt, 3, green(), 2, cv::LineTypes::FILLED);
    }

    if (app.right_idx >= 0 && app.right_idx < app.keypoints2.size())
    {
        cv::Point2f offset(app.image1.cols, 0);
        cv::circle(app.out_image, app.keypoints2[app.right_idx].pt + offset, 3, green(), 2, cv::LineTypes::FILLED);
    }
}

void draw_similarity_circles(App& app)
{
    int         radius    = 4;
    int         thickness = 3;
    cv::Point2f offset    = cv::Point2f(app.image1.cols, 0);

    if (app.next_sel_wheel > app.next_matches.size() - 1)
    {
        std::cout << "ERROR in draw_similarity_circles: next_sel_wheel larger than next_matches.size()" << std::endl;
        return;
    }

    if (app.last_click_was_left)
    {
        //mark selected on the left
        cv::circle(app.out_image, app.keypoints1[app.left_idx].pt, radius, red(), thickness);

        //draw matches on the right
        for (auto it = app.next_matches.begin(); it != app.next_matches.end(); ++it)
        {
            cv::circle(app.out_image, app.keypoints2[it->idx].pt + offset, radius, it->color, thickness);
        }

        cv::line(app.out_image,
                 app.keypoints1[app.left_idx].pt,
                 app.keypoints2[app.next_matches[app.next_sel_wheel].idx].pt + offset,
                 app.next_matches[app.next_sel_wheel].color,
                 2);
    }
    else
    {
        //mark selected on the right
        cv::circle(app.out_image, app.keypoints2[app.right_idx].pt + offset, radius, red(), thickness);

        //draw matches on the left
        for (auto it = app.next_matches.begin(); it != app.next_matches.end(); ++it)
        {
            cv::circle(app.out_image, app.keypoints1[it->idx].pt, radius, it->color, thickness);
        }

        cv::line(app.out_image,
                 app.keypoints2[app.right_idx].pt + offset,
                 app.keypoints1[app.next_matches[app.next_sel_wheel].idx].pt,
                 app.next_matches[app.next_sel_wheel].color,
                 2);
    }
}

void draw_match_line(App& app, int matchIndex1, int matchIndex2, cv::Scalar color)
{
    if (matchIndex1 >= app.keypoints1.size() || matchIndex2 >= app.keypoints2.size())
    {
        std::cout << "WARNING in function draw_match_line: indices out of range!" << std::endl;
        return;
    }

    cv::Point2f offset(app.image1.cols, 0);
    cv::circle(app.out_image, app.keypoints1[matchIndex1].pt, 3, color, 1);
    cv::circle(app.out_image, app.keypoints2[matchIndex2].pt + offset, 3, color, 1);
    cv::line(app.out_image,
             app.keypoints2[matchIndex2].pt + offset,
             app.keypoints1[matchIndex1].pt,
             color,
             2);
}

//draw all keypoints and all matches
void draw_matches_lines(App& app, const cv::Scalar color)
{
    cv::Point2f offset(app.image1.cols, 0);
    for (int i = 0; i < app.matching_2_1.size(); i++)
    {
        if (app.matching_2_1[i] >= 0)
        {
            cv::circle(app.out_image, app.keypoints2[i].pt + offset, 3, color, 1);
            cv::circle(app.out_image, app.keypoints1[app.matching_2_1[i]].pt, 3, color, 1);
            cv::line(app.out_image, app.keypoints2[i].pt + offset, app.keypoints1[app.matching_2_1[i]].pt, color, 2);
        }
    }
}

//void draw_by_similarity(App& app)
//{
//    cv::hconcat(app.image1, app.image2, app.out_image);
//
//    for (int i = 0; i < app.ordered_keypoints1.size(); i++)
//    {
//        cv::circle(app.out_image, app.ordered_keypoints1[i].pt, 3, app.kp1_colors[i], 1);
//        if (app.ordered_keypoints1[i].response < 0.5)
//        {
//            cv::circle(app.out_image, app.ordered_keypoints1[i].pt, 15, app.kp1_colors[i], 3);
//        }
//    }
//    for (int i = 0; i < app.keypoints2.size(); i++)
//    {
//        cv::circle(app.out_image, app.ordered_keypoints2[i].pt + cv::Point2f(app.image1.cols, 0), 3, app.kp2_colors[i], 1);
//        if (app.ordered_keypoints2[i].response < 0.5)
//        {
//            cv::circle(app.out_image, app.ordered_keypoints2[i].pt + cv::Point2f(app.image1.cols, 0), 15, app.kp2_colors[i], 3);
//        }
//    }
//}

void draw_main(App& app)
{
    cv::Mat out;
    cv::copyMakeBorder(app.out_image, out, 0, 100, 0, 0, cv::BORDER_CONSTANT, 0);
    cv::Point pos(30, app.out_image.rows + 30);
    int       font      = cv::FONT_HERSHEY_PLAIN;
    float     fontScale = 1.0;
    int       thick     = 1;

    std::string text = app_inspection_mode_text(app);
    text             = std::to_string((int)app.inspectionMode - 48) + ": " + text;
    if (text.length() > 0)
    {
        std::vector<std::string> strs = str_split(text);
        for (int i = 0; i < strs.size(); i++)
        {
            cv::putText(out, strs[i], cv::Point(pos.x, pos.y + 20 + 20 * i), font, fontScale, cv::Scalar(255, 255, 255), thick, cv::LINE_AA);
        }
    }

    switch (app.method)
    {
        case STOCK_ORBSLAM:
            cv::putText(out, "ORB keypoint, ORB descrptor", pos, font, fontScale, cv::Scalar(255, 255, 255), thick, cv::LINE_AA);
            break;

        case STOCK_ORBSLAM_CLAHE:
            cv::putText(out, "ORB keypoint, ORB descrptor CLAHE", pos, font, fontScale, cv::Scalar(255, 255, 255), thick, cv::LINE_AA);
            break;

        case TILDE_BRIEF:
            cv::putText(out, "TILDE keypoint, BRIEF descrptor", pos, font, fontScale, cv::Scalar(255, 255, 255), thick, cv::LINE_AA);
            break;

        case TILDE_BRIEF_CLAHE:
            cv::putText(out, "TILDE keypoint, BRIEF descrptor CLAHE", pos, font, fontScale, cv::Scalar(255, 255, 255), thick, cv::LINE_AA);
            break;

        case SURF_BRIEF:
            cv::putText(out, "SURF keypoint, BRIEF descrptor", pos, font, fontScale, cv::Scalar(255, 255, 255), thick, cv::LINE_AA);
            break;

        case SURF_BRIEF_CLAHE:
            cv::putText(out, "SURF keypoint, BRIEF descrptor CLAHE", pos, font, fontScale, cv::Scalar(255, 255, 255), thick, cv::LINE_AA);
            break;

        case SURF_ORB:
            cv::putText(out, "SURF keypoint, ORB descrptor", pos, font, fontScale, cv::Scalar(255, 255, 255), thick, cv::LINE_AA);
            break;

        case SURF_ORB_CLAHE:
            cv::putText(out, "SURF keypoint, ORB descrptor CLAHE", pos, font, fontScale, cv::Scalar(255, 255, 255), thick, cv::LINE_AA);
            break;
    }

    imshow(app.name, out);
}
