#include "gui_tools.h"

cv::Mat draw_closeup(cv::Mat& image, cv::Point2f& pt, std::string text)
{
    cv::Mat          out;
    cv::Mat          closeup;
    std::vector<int> umax;

    init_patch(umax);
    cv::Mat patch = extract_patch(image, pt);
    cv::resize(patch, closeup, cv::Size(500, 500), cv::INTER_NEAREST);
    cv::copyMakeBorder(closeup, out, 0, 200, 0, 0, cv::BORDER_CONSTANT, 0);

    if (text.length() > 0)
    {
        std::vector<std::string> strs = str_split(text);
        for (int i = 0; i < strs.size(); i++)
        {
            cv::putText(out, strs[i], cv::Point(10, closeup.rows + 30 + 20 * i), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }

    return out;
}

void draw_closeup_similarity(App& app)
{
    int   idx1 = -1, idx2 = -1;
    float distance = -1;

    //estimate wheel selected
    const App::NextMatch& nextMatch = app.next_matches[app.next_sel_wheel];
    if (app.last_click_was_left)
    {
        idx1     = app.left_idx;
        idx2     = nextMatch.idx;
        distance = nextMatch.distance;
    }
    else
    {
        idx1     = nextMatch.idx;
        idx2     = app.right_idx;
        distance = nextMatch.distance;
    }

    //draw left closeup
    {
        cv::KeyPoint&     kpt1 = app.keypoints1[idx1];
        std::stringstream ss;
        ss << "Point idx: " << idx1 << std::endl;
        ss << "Octave: " << kpt1.octave << std::endl;
        ss << "size: " << kpt1.size << std::endl;
        ss << "Angle: " << kpt1.angle << std::endl;
        ss << "Distance to right index " << idx2 << " is " << distance << std::endl;

        cv::Mat out;
        if (app.method == STOCK_ORBSLAM)
        {
            out = draw_closeup(app.image1_pyramid[kpt1.octave], kpt1.pt, ss.str());
        }
        else
        {
            out = draw_closeup(app.image1, kpt1.pt, ss.str());
        }
        imshow(app.closeup_left, out);
    }

    //draw right closeup
    {
        cv::KeyPoint&     kpt2 = app.keypoints2[idx2];
        std::stringstream ss;
        ss << "Point idx: " << idx2 << std::endl;
        ss << "Octave: " << kpt2.octave << std::endl;
        ss << "size: " << kpt2.size << std::endl;
        ss << "Angle: " << kpt2.angle << std::endl;
        ss << "Distance to left index " << idx1 << " is " << distance << std::endl;

        cv::Mat out;
        if (app.method == STOCK_ORBSLAM)
        {
            out = draw_closeup(app.image2_pyramid[kpt2.octave], kpt2.pt, ss.str());
        }
        else
        {
            out = draw_closeup(app.image2, kpt2.pt, ss.str());
        }
        imshow(app.closeup_right, out);
    }
}

void draw_closeup_right(App& app, bool calcDistSelected)
{
    if (app.right_idx < 0)
    {
        cv::destroyWindow("closeup right");
        return;
    }

    std::stringstream ss;
    ss << "Point idx: " << app.right_idx << std::endl;
    ss << "Octave: " << app.keypoints2[app.right_idx].octave << std::endl;
    ss << "size: " << app.keypoints2[app.right_idx].size << std::endl;
    ss << "Angle: " << app.keypoints2[app.right_idx].angle << std::endl;
    if (app.matching_2_1[app.right_idx] >= 0)
    {
        ss << "Distance to best match with index " << app.matching_2_1[app.right_idx];
        ss << " is " << hamming_distance(app.descs1[app.matching_2_1[app.right_idx]], app.descs2[app.right_idx]) << std::endl;
    }

    //calculate distance between selected features of selected indices
    if (calcDistSelected)
    {
        if (app.right_idx >= 0 && app.right_idx < app.descs2.size() &&
            app.left_idx >= 0 && app.left_idx < app.descs1.size())
        {
            ss << "Distance to left index " << app.left_idx << " is " << hamming_distance(app.descs1[app.left_idx], app.descs2[app.right_idx]) << std::endl;
        }
    }

    cv::Mat out;
    if (app.method == STOCK_ORBSLAM)
    {
        out = draw_closeup(app.image2_pyramid[app.keypoints2[app.right_idx].octave], app.keypoints2[app.right_idx].pt, ss.str());
    }
    else
    {
        out = draw_closeup(app.image2, app.keypoints2[app.right_idx].pt, ss.str());
    }
    imshow(app.closeup_right, out);
}

void draw_closeup_left(App& app, bool calcDistSelected)
{
    if (app.left_idx < 0)
    {
        cv::destroyWindow(app.closeup_left);
        return;
    }

    std::stringstream ss;
    ss << "Point idx: " << app.left_idx << std::endl;
    ss << "octave: " << app.keypoints1[app.left_idx].octave << std::endl;
    ss << "size: " << app.keypoints1[app.left_idx].size << std::endl;
    ss << "Angle: " << app.keypoints1[app.left_idx].angle << std::endl;
    if (app.matching_1_2[app.left_idx] >= 0)
    {
        ss << "Distance to best match with index " << app.matching_1_2[app.left_idx];
        ss << " is " << hamming_distance(app.descs1[app.left_idx], app.descs2[app.matching_1_2[app.left_idx]]) << std::endl;
    }

    //calculate distance between selected features of selected indices
    if (calcDistSelected)
    {
        if (app.right_idx >= 0 && app.right_idx < app.descs2.size() &&
            app.left_idx >= 0 && app.left_idx < app.descs1.size())
        {
            ss << "Distance to right index " << app.right_idx << " is " << hamming_distance(app.descs1[app.left_idx], app.descs2[app.right_idx]) << std::endl;
        }
    }

    cv::Mat out;
    if (app.method == STOCK_ORBSLAM)
    {
        out = draw_closeup(app.image1_pyramid[app.keypoints1[app.left_idx].octave], app.keypoints1[app.left_idx].pt, ss.str());
    }
    else
    {
        out = draw_closeup(app.image1, app.keypoints1[app.left_idx].pt, ss.str());
    }

    imshow(app.closeup_left, out);
}

void match_drawing_all(App& app)
{
    cv::destroyWindow("closeup left");
    cv::destroyWindow("closeup right");

    draw_concat_images(app);
    draw_all_keypoins(app, blue());
    draw_matches_lines(app, red());
    draw_main(app);
}

void match_drawing_single(int x, int y, int flags, App& app)
{
    int idx1 = 0, idx2 = 0;
    //input for image displayed on the right
    if (x > app.image1.cols)
    {
        idx2 = select_closest_keypoint(app.keypoints2, app.matching_2_1, x - app.image1.cols, y);
        if (idx2 < 0)
        {
            std::cout << "INFO in function match_drawing_single: no feature selected" << std::endl;
            return;
        }

        idx1 = app.matching_2_1[idx2];

        app.left_idx  = idx1;
        app.right_idx = idx2;
    }
    else //input for image displayed on the left
    {
        idx1 = select_closest_keypoint(app.keypoints1, app.matching_1_2, x, y);
        if (idx1 < 0)
        {
            std::cout << "INFO in function match_drawing_single: no feature selected" << std::endl;
            return;
        }

        idx2 = app.matching_1_2[idx1];

        app.left_idx  = idx1;
        app.right_idx = idx2;
    }

    draw_closeup_left(app, false);
    draw_closeup_right(app, false);

    draw_concat_images(app);
    draw_all_keypoins(app, blue());
    draw_matched_keypoints(app, red());
    draw_match_line(app, idx1, idx2, red());
    draw_main(app);
}

//find n next best matches in other image.
//Attention: n can be reduced if not enough matches
std::vector<App::NextMatch> select_n_closest_features(const std::vector<cv::KeyPoint>& keypoints,
                                                      std::vector<Descriptor>&         descs,
                                                      Descriptor&                      refDesc,
                                                      int&                             n)
{
    std::vector<App::NextMatch> nextMatches;
    if (!keypoints.size())
        return nextMatches;

    //Attention: n can be reduced if not enough matches
    n = (keypoints.size() < n) ? keypoints.size() : n;

    float max_dist = 0;
    float min_dist = 0;

    //find next n matches
    std::vector<App::NextMatch> matches(keypoints.size());
    {
        //calculate distance of
        for (int i = 0; i < keypoints.size(); i++)
        {
            matches[i].distance = hamming_distance(refDesc, descs[i]);
            matches[i].idx      = i;
        }

        //extract n best
        std::sort(matches.begin(), matches.end());
        std::copy(matches.begin(), matches.begin() + n, std::back_inserter(nextMatches));
    }

    //calculate color value
    float min = nextMatches.front().distance;
    float max = nextMatches.back().distance;
    float w   = max - min;

    cv::Mat valImg(n, 1, CV_8UC1);
    for (int i = 0; i < n; ++i)
    {
        float frag             = (nextMatches[i].distance - min) / w;
        uchar val              = (uchar)(frag * 255.f);
        valImg.at<uchar>(i, 0) = val;
    }

    cv::Mat colImg;
    cv::applyColorMap(valImg, colImg, cv::ColormapTypes::COLORMAP_AUTUMN);
    for (int i = 0; i < n; ++i)
    {
        nextMatches[i].color = colImg.at<cv::Vec3b>(i, 0);
    }

    return nextMatches;
}

void matched_point_similarity(int x, int y, int flags, App& app)
{
    //reset mousewheel selected index after on click
    app.next_sel_wheel = 0;

    if (x > app.image1.cols) //right image
    {
        app.last_click_was_left = false;
        //find next keypoint to click in right image
        app.right_idx = select_closest_keypoint(app.keypoints2, x - app.image1.cols, y);
        //find next descriptors in left image
        app.next_matches = select_n_closest_features(app.keypoints1, app.descs1, app.descs2[app.right_idx], app.num_next_matches);
    }
    else //left image
    {
        app.last_click_was_left = true;
        app.left_idx            = select_closest_keypoint(app.keypoints1, x, y);
        //find next descriptors in left image
        app.next_matches = select_n_closest_features(app.keypoints2, app.descs2, app.descs1[app.left_idx], app.num_next_matches);
    }

    draw_closeup_similarity(app);

    draw_concat_images(app);
    draw_all_keypoins(app, blue());
    draw_matched_keypoints(app, red());
    draw_similarity_circles(app);
    draw_main(app);
}

void any_keypoint_comparison(int x, int y, int flags, App& app)
{
    app.ordered_keypoints1 = app.keypoints1;
    app.ordered_keypoints2 = app.keypoints2;

    if (x > app.image1.cols)
    {
        std::vector<int> idxs2 = select_closest_features(app.ordered_keypoints2, app.select_radius, x - app.image1.cols, y);
        int              idx2;

        if (idxs2.size() > 1)
        {
            if (app.local_idx >= idxs2.size())
                app.local_idx = 0;

            idx2 = idxs2[app.local_idx++];
        }
        else if (idxs2.size() == 1)
        {
            idx2 = idxs2[0];
        }
        else
        {
            idx2 = select_closest_keypoint(app.ordered_keypoints2, x - app.image1.cols, y);
        }

        app.right_idx = idx2;
    }
    else
    {
        std::vector<int> idxs1 = select_closest_features(app.ordered_keypoints1, app.select_radius, x, y);
        int              idx1;
        if (idxs1.size() > 1)
        {
            if (app.local_idx >= idxs1.size())
                app.local_idx = 0;

            idx1 = idxs1[app.local_idx++];
        }
        else if (idxs1.size() == 1)
        {
            idx1 = idxs1[0];
        }
        else
        {
            idx1 = select_closest_keypoint(app.ordered_keypoints1, x, y);
        }

        app.left_idx = idx1;
    }

    draw_closeup_left(app, true);
    draw_closeup_right(app, true);

    draw_concat_images(app);
    draw_all_keypoins(app, blue());
    draw_matched_keypoints(app, red());
    draw_selected_keypoints(app);

    draw_main(app);
}

void update_inspection(App& app)
{
    const int x     = app.mouse_pos.x;
    const int y     = app.mouse_pos.y;
    const int flags = app.keyboard_flags;

    switch (app.inspectionMode)
    {
        case InspectionMode::MATCH_DRAWING_ALL:
            match_drawing_all(app);
            break;
        case InspectionMode::MATCH_DRAWING_SINGLE:
            match_drawing_single(x, y, flags, app);
            break;
        case InspectionMode::MATCHED_POINT_SIMILIARITY:
            matched_point_similarity(x, y, flags, app);
            break;
        case InspectionMode::ANY_KEYPOINT_COMPARISON:
            any_keypoint_comparison(x, y, flags, app);
            break;
        default:
            throw std::runtime_error("Unknown InspectionMode");
    }
}

void mouse_button_left(int x, int y, int flags, App& app)
{
    //if the mouse position has changed we reset the local_idx which is used to iterate close lying points
    if (x != app.mouse_pos.x && y != app.mouse_pos.y)
        app.local_idx = 0;

    app.mouse_pos.x    = x;
    app.mouse_pos.y    = y;
    app.keyboard_flags = flags;
    update_inspection(app);
}

void on_mouse_wheel(App& app, int flags)
{
    int val = cv::getMouseWheelDelta(flags);
    if (val > 0)
        app.next_sel_wheel = ++app.next_sel_wheel % app.num_next_matches;
    else
        app.next_sel_wheel = --app.next_sel_wheel < 0 ? app.num_next_matches - 1 : app.next_sel_wheel;

    switch (app.inspectionMode)
    {
        case InspectionMode::MATCH_DRAWING_ALL:
            break;
        case InspectionMode::MATCH_DRAWING_SINGLE:
            break;
        case InspectionMode::MATCHED_POINT_SIMILIARITY:
            draw_closeup_similarity(app);
            draw_concat_images(app);
            draw_all_keypoins(app, blue());
            draw_matched_keypoints(app, red());
            draw_similarity_circles(app);
            draw_main(app);

            break;
        case InspectionMode::ANY_KEYPOINT_COMPARISON:
            break;
        default:
            throw std::runtime_error("Unknown InspectionMode");
    }
}

void main_mouse_events(int event, int x, int y, int flags, void* userdata)
{
    App& app = *(App*)userdata;

    switch (event)
    {
        case cv::EVENT_LBUTTONDOWN: {
            mouse_button_left(x, y, flags, app);
            break;
        }
        case cv::EVENT_MOUSEWHEEL: {
            on_mouse_wheel(app, flags);
            break;
        }
    }
}

void print_help()
{
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Usage of Best Tool Ever Made" << std::endl;
    std::cout << "Space:    Change detection and description method" << std::endl;
    std::cout << "1-4:      Change inspection mode" << std::endl;
    std::cout << "LMB:      Select closest match" << std::endl;
    std::cout << "M-Wheel:  Select between next matches" << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
}

bool update_inspection_mode(const int key, App& app)
{
    InspectionMode newMode = (InspectionMode)key;
    if (newMode < InspectionMode::END)
    {
        app.inspectionMode = newMode;
        return true;
    }
    else
    {
        std::cout << "INFO: update_inspection_mode: unused key for mode selection" << std::endl;
        return false;
    }
}

void update_detection(App& app)
{
    app_reset(app);
    app_prepare(app);
}

void start_gui(App& app)
{
    cv::namedWindow(app.name, 1);
    cv::setMouseCallback(app.name, main_mouse_events, &app);
    std::cout << "Welcome to best tool ever made!" << std::endl;
    print_help();

    //-----------------------------------------------------------------------------
    //config
    app.method           = SURF_BRIEF;
    app.inspectionMode   = InspectionMode::MATCHED_POINT_SIMILIARITY;
    app.num_next_matches = 10;

    app.clahe = cv::createCLAHE();
    app.clahe->setClipLimit(2.0);
    app.clahe->setTilesGridSize(cv::Size(8,8));
    //-----------------------------------------------------------------------------

    update_detection(app);
    update_inspection(app);

    for (;;)
    {
        int retval = cv::waitKey(0);
        if (retval == ' ') //change extraction method with space
        {
            app_next_method(app);
            update_detection(app);
            update_inspection(app);
        }
        else if (retval > 47 && retval < 58) //numbers pressed for inspection mode change
        {
            if (update_inspection_mode(retval, app))
                update_inspection(app);
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
