#include "gui_tools.h"

cv::Mat draw_closeup(cv::Mat& image, cv::KeyPoint& kp, std::string text)
{
    cv::Mat          out;
    cv::Mat          closeup;
    std::vector<int> umax;

    init_patch(umax);
    cv::Mat patch = extract_patch(image, kp);
    cv::resize(patch, closeup, cv::Size(500, 500), cv::INTER_NEAREST);
    cv::copyMakeBorder(closeup, out, 0, 200, 0, 0, cv::BORDER_CONSTANT, 0);

    if (text.length() > 0)
    {
        std::vector<std::string> strs = str_split(text);
        for (int i = 0; i < strs.size(); i++)
        {
            cv::putText(out, strs[i], cv::Point(10, closeup.rows + 30 + 20 * i), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
    }

    return out;
}

void draw_closeup_right(App& app, int idx)
{
    app.right_idx = idx;
    std::stringstream ss;
    ss << "Point idx: " << idx << std::endl;
    ss << "Octave: " << app.keypoints2[idx].octave << std::endl;
    ss << "size: " << app.keypoints2[idx].size << std::endl;
    ss << "Angle: " << app.keypoints2[idx].angle << std::endl;
    if (app.matching_2_1[idx] >= 0)
        ss << "Has matching to " << app.matching_2_1[idx] << std::endl;

    int    prop = cv::WindowPropertyFlags::WND_PROP_VISIBLE;
    double ret  = cv::getWindowProperty(app.closeup_right, prop);
    if (ret >= 0)
        cv::getWindowProperty(app.closeup_left, prop);
    if (prop >= 0)
        ss << "Distance " << hamming_distance(app.descs1[app.left_idx], app.descs2[app.right_idx]) << std::endl;

    cv::Mat out;
    if (app.method == STOCK_ORBSLAM)
    {
        out = draw_closeup(app.image2_pyramid[app.keypoints2[idx].octave], app.keypoints2[idx], ss.str());
    }
    else
    {
        out = draw_closeup(app.image2, app.keypoints2[idx], ss.str());
    }
    imshow(app.closeup_right, out);
}

void draw_closeup_left(App& app, int idx)
{
    app.left_idx = idx;
    std::stringstream ss;
    ss << "Point idx: " << idx << std::endl;
    ss << "octave: " << app.keypoints1[idx].octave << std::endl;
    ss << "size: " << app.keypoints1[idx].size << std::endl;
    ss << "Angle: " << app.keypoints1[idx].angle << std::endl;
    if (app.matching_1_2[idx] >= 0)
        ss << "Has matching to " << app.matching_1_2[idx] << std::endl;

    int    prop = cv::WindowPropertyFlags::WND_PROP_VISIBLE;
    double ret  = cv::getWindowProperty(app.closeup_right, prop);
    if (ret >= 0)
        ss << "Distance " << hamming_distance(app.descs1[app.left_idx], app.descs2[app.right_idx]) << std::endl;

    cv::Mat out;
    if (app.method == STOCK_ORBSLAM)
    {
        out = draw_closeup(app.image1_pyramid[app.keypoints1[idx].octave], app.keypoints1[idx], ss.str());
    }
    else
    {
        out = draw_closeup(app.image1, app.keypoints1[idx], ss.str());
    }

    imshow(app.closeup_left, out);
}

void draw_main(App& app)
{
    cv::Mat out;
    cv::copyMakeBorder(app.out_image, out, 0, 100, 0, 0, cv::BORDER_CONSTANT, 0);
    cv::Point pos(30, app.out_image.rows + 30);

    std::string text = app.inspection_mode_text();

    if (text.length() > 0)
    {
        std::vector<std::string> strs = str_split(text);
        for (int i = 0; i < strs.size(); i++)
        {
            cv::putText(out, strs[i], cv::Point(pos.x, pos.y + 20 + 20 * i), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
    }

    switch (app.method)
    {
        case STOCK_ORBSLAM:
            cv::putText(out, "ORB keypoint, ORB descrptor", pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            break;

        case TILDE_BRIEF:
            cv::putText(out, "TILDE keypoint, BRIEF descrptor", pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            break;

        case SURF_BRIEF:
            cv::putText(out, "SURF keypoint, BRIEF descrptor", pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            break;

        case SURF_ORB:
            cv::putText(out, "SURF keypoint, ORB descrptor", pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            break;
    }

    imshow(app.name, out);
}

bool sort_fct(cv::KeyPoint& p1, cv::KeyPoint& p2)
{
    return p1.response > p2.response;
}

void match_drawing(int x, int y, int flags, App* app)
{
    reset_color(app->kp1_colors, blue());
    reset_color(app->kp2_colors, blue());

    //input for image displayed on the right
    if (x > app->image1.cols)
    {
        int idx2 = select_closest_feature(app->keypoints2, app->matching_2_1, x - app->image1.cols, y);
        if (idx2 < 0) { return; }

        int idx1 = app->matching_2_1[idx2];
        app->poi = app->keypoints2[idx2].pt;

        if ((flags & cv::EVENT_FLAG_CTRLKEY) && cv::EVENT_FLAG_CTRLKEY)
        {
            app->left_idx  = idx1;
            app->right_idx = idx2;
            draw_closeup_left(*app, idx1);
            draw_closeup_right(*app, idx2);
        }
        app->kp1_colors[idx1] = red();
        app->kp2_colors[idx2] = red();
        draw_match_line(*app, idx1, idx2);
    }
    else //input for image displayed on the left
    {
        int idx1 = select_closest_feature(app->keypoints1, app->matching_1_2, x, y);
        if (idx1 < 0) { return; }

        app->poi = app->keypoints1[idx1].pt;
        int idx2 = app->matching_1_2[idx1];

        if ((flags & cv::EVENT_FLAG_CTRLKEY) && cv::EVENT_FLAG_CTRLKEY)
        {
            app->left_idx  = idx1;
            app->right_idx = idx2;
            draw_closeup_left(*app, idx1);
            draw_closeup_right(*app, idx2);
        }
        app->kp1_colors[idx1] = red();
        app->kp2_colors[idx2] = red();
        draw_match_line(*app, idx1, idx2);
    }

    draw_main(*app);
}

void matched_point_similarity(int x, int y, int flags, App* app)
{
    reset_similarity(app->keypoints1);
    reset_similarity(app->keypoints2);
    reset_color(app->kp1_colors, blue());
    reset_color(app->kp2_colors, blue());

    app->ordered_keypoints1 = app->keypoints1;
    app->ordered_keypoints2 = app->keypoints2;

    if (x > app->image1.cols)
    {
        std::vector<int> idxs2 = select_closest_features(app->ordered_keypoints2, app->select_radius, x - app->image1.cols, y);
        int              idx2;

        if (idxs2.size() > 1)
        {
            if (app->local_idx >= idxs2.size())
                app->local_idx = 0;

            idx2 = idxs2[app->local_idx++];
        }
        else if (idxs2.size() == 1)
        {
            idx2 = idxs2[0];
        }
        else
        {
            idx2 = select_closest_feature(app->ordered_keypoints2, x - app->image1.cols, y);
        }

        draw_closeup_right(*app, idx2);
        app->kp2_colors[idx2] = red();

        compute_similarity(app->ordered_keypoints1, app->descs1, app->descs2[idx2]);
        sort(app->ordered_keypoints1.begin(), app->ordered_keypoints1.end(), sort_fct);
        set_color_by_value(app->kp1_colors, app->ordered_keypoints1);
    }
    else
    {
        std::vector<int> idxs1 = select_closest_features(app->ordered_keypoints1, app->select_radius, x, y);
        int              idx1;
        if (idxs1.size() > 1)
        {
            if (app->local_idx >= idxs1.size())
                app->local_idx = 0;

            idx1 = idxs1[app->local_idx++];
        }
        else if (idxs1.size() == 1)
        {
            idx1 = idxs1[0];
        }
        else
        {
            idx1 = select_closest_feature(app->ordered_keypoints1, x, y);
        }

        draw_closeup_left(*app, idx1);
        app->kp1_colors[idx1] = red();

        compute_similarity(app->ordered_keypoints2, app->descs2, app->descs1[idx1]);
        sort(app->ordered_keypoints2.begin(), app->ordered_keypoints2.end(), sort_fct);
        set_color_by_value(app->kp2_colors, app->ordered_keypoints2);
    }

    draw_by_similarity(*app);
    draw_main(*app);
}

void any_keypoint_comparison(int x, int y, int flags, App* app)
{
}

void any_pixel_comparison(int x, int y, int flags, App* app)
{
}

void mouse_button_left(int x, int y, int flags, App* app)
{
    switch (app->inspectionMode)
    {
        case InspectionMode::MATCH_DRAWING:
            match_drawing(x, y, flags, app);
            break;
        case InspectionMode::MATCHED_POINT_SIMILIARITY:
            matched_point_similarity(x, y, flags, app);
            break;
        case InspectionMode::ANY_KEYPOINT_COMPARISON:
            any_keypoint_comparison(x, y, flags, app);
            break;
        case InspectionMode::ANY_PIXEL_COMPARISON:
            break;
            any_pixel_comparison(x, y, flags, app);
            break;
        default:
            throw std::runtime_error("Unknown InspectionMode");
    }
}

void main_mouse_events(int event, int x, int y, int flags, void* userdata)
{
    App* app = (App*)userdata;

    switch (event)
    {
        case cv::EVENT_LBUTTONDOWN: {
            mouse_button_left(x, y, flags, app);
            break;
        }
    }
}

void print_help()
{
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Usage of Best Tool Ever Made" << std::endl;
    std::cout << "Space:    change detection and description method" << std::endl;
    std::cout << "1:        Change inspection mode to 'draw matching'" << std::endl;
    std::cout << "2:        Change inspection mode to 'matched points similarity'" << std::endl;
    std::cout << "3:        Change inspection mode to 'any keypoint comparison'" << std::endl;
    std::cout << "4:        Change inspection mode to 'any pixel comparison'" << std::endl;
    std::cout << "LMB:      Select closest match and highlight in red" << std::endl;
    std::cout << "Ctrl+LMB: Additionally update closeup windows of closest matching pair" << std::endl;
    //std::cout << "RMB:      Select closest keypoint and visualize closest matches in other image." << std::endl;
    //std::cout << "          Show closeup window of closest match." << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
}

void update_inspection_mode(const int key, App& app)
{
    InspectionMode newMode = (InspectionMode)key;
    if (newMode < InspectionMode::END)
    {
        app.inspectionMode = newMode;
    }
    else
        std::cout << "INFO: update_inspection_mode: unused key for mode selection" << std::endl;
}

void update_detection(App& app)
{
    app_reset(app);
    app_prepare(app);

    init_color(app.kp1_colors, app.keypoints1.size());
    init_color(app.kp2_colors, app.keypoints2.size());

    draw_matches_lines(app);
    draw_main(app);
}

void start_gui(App& app)
{
    cv::namedWindow(app.name, 1);
    cv::setMouseCallback(app.name, main_mouse_events, &app);
    std::cout << "Welcome to best tool ever made!" << std::endl;
    print_help();

    init_color(app.kp1_colors, app.keypoints1.size());
    init_color(app.kp2_colors, app.keypoints2.size());

    //in the first run use default detection method
    app.method = SURF_BRIEF;
    update_detection(app);

    for (;;)
    {
        int retval = cv::waitKey(0);
        if (retval == ' ') //change extraction method with space
        {
            app_next_method(app);
            update_detection(app);
        }
        else if (retval > 47 && retval < 58) //numbers pressed for inspection mode change
        {
            update_inspection_mode(retval, app);
            update_detection(app);
        }
        else if (retval == 105 || retval == 104) // i for info or h for help
        {
            print_help();
        }
        else // else show error message and print help
        {
            std::cout << "unknown keyboard input" << std::endl;
            print_help();
        }
    }
    cv::destroyAllWindows();
}
