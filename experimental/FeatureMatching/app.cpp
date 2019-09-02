#include "app.h"
#include "convert.h"
#include "ExtractKeypoints.h"
#include "orb_descriptor.h"
#include "brief_descriptor.h"
#include "matching.h"

void orb_descriptors_orb_keypoints(App &app)
{
    cv::Mat grayscaleImg;
    std::vector<std::vector<cv::KeyPoint>> all_keypoints;
    std::vector<std::vector<Descriptor>> all_desc;

    //Image 1
    grayscaleImg = rgb_to_grayscale(app.image1);

#if EQUALIZE_HIST == 1
    equalizeHist(grayscaleImg, grayscaleImg);
#endif

    build_pyramid(app.image1_pyramid, grayscaleImg, app.pyramid_param);
    KPExtractOrbSlam(all_keypoints, app.image1_pyramid, app.pyramid_param);
    ComputeORBDescriptors(all_desc, app.image1_pyramid, app.pyramid_param, all_keypoints);

    flatten_keypoints(app.keypoints1, all_keypoints, app.pyramid_param);
    flatten_decriptors(app.descs1, all_desc, app.pyramid_param);


    //Image 2
    all_keypoints.clear();
    all_desc.clear();

    grayscaleImg = rgb_to_grayscale(app.image2);

#if EQUALIZE_HIST == 1
    equalizeHist(grayscaleImg, grayscaleImg);
#endif

    build_pyramid(app.image2_pyramid, grayscaleImg, app.pyramid_param);
    KPExtractOrbSlam(all_keypoints, app.image2_pyramid, app.pyramid_param);
    ComputeORBDescriptors(all_desc, app.image2_pyramid, app.pyramid_param, all_keypoints);

    flatten_keypoints(app.keypoints2, all_keypoints, app.pyramid_param);
    flatten_decriptors(app.descs2, all_desc, app.pyramid_param);


    match_keypoints_1(app.matching_2_1, app.keypoints1, app.descs1, app.keypoints2, app.descs2, true);
}

void brief_descriptors_tilde_keypoints(App &app)
{
    cv::Mat grayscaleImg1;
    cv::Mat grayscaleImg2;
    app.image1_pyramid.clear();
    app.image2_pyramid.clear();
    
    grayscaleImg1 = rgb_to_grayscale(app.image1);
    app.image1_pyramid.push_back(grayscaleImg1.clone());
    KPExtractTILDE(app.keypoints1, app.image1);
    ComputeBRIEFDescriptors(app.descs1, grayscaleImg1, app.keypoints1);

    //Image 2

    grayscaleImg2 = rgb_to_grayscale(app.image2);
    app.image2_pyramid.push_back(grayscaleImg2.clone());
    KPExtractTILDE(app.keypoints2, app.image2);
    ComputeBRIEFDescriptors(app.descs2, grayscaleImg2, app.keypoints2);

    match_keypoints_1(app.matching_2_1, app.keypoints1, app.descs1, app.keypoints2, app.descs2, false);
}

void orb_descriptors_surf_keypoints(App &app)
{
    cv::Mat grayscaleImg1;
    cv::Mat grayscaleImg2;
    app.image1_pyramid.clear();
    app.image2_pyramid.clear();
    
    grayscaleImg1 = rgb_to_grayscale(app.image1);
    app.image1_pyramid.push_back(grayscaleImg1.clone());
    KPExtractSURF(app.keypoints1, app.image1);
    ComputeORBDescriptors(app.descs1, grayscaleImg1, app.keypoints1);

    //Image 2

    grayscaleImg2 = rgb_to_grayscale(app.image2);
    app.image2_pyramid.push_back(grayscaleImg2.clone());
    KPExtractSURF(app.keypoints2, app.image2);
    ComputeORBDescriptors(app.descs2, grayscaleImg2, app.keypoints2);


    match_keypoints_1(app.matching_2_1, app.keypoints1, app.descs1, app.keypoints2, app.descs2, true);
}

void brief_descriptors_surf_keypoints(App &app)
{
    cv::Mat grayscaleImg1;
    cv::Mat grayscaleImg2;
    app.image1_pyramid.clear();
    app.image2_pyramid.clear();
    
    grayscaleImg1 = rgb_to_grayscale(app.image1);
    app.image1_pyramid.push_back(grayscaleImg1.clone());
    KPExtractSURF(app.keypoints1, app.image1);
    ComputeBRIEFDescriptors(app.descs1, grayscaleImg1, app.keypoints1);

    //Image 2

    grayscaleImg2 = rgb_to_grayscale(app.image2);
    app.image2_pyramid.push_back(grayscaleImg2.clone());
    KPExtractSURF(app.keypoints2, app.image2);
    ComputeBRIEFDescriptors(app.descs2, grayscaleImg2, app.keypoints2);

    match_keypoints_1(app.matching_2_1, app.keypoints1, app.descs1, app.keypoints2, app.descs2, false);
}

void app_next_method(App &app)
{
    app.method = (app.method+1) % END_METHOD;
}

void app_reset(App &app)
{
    app.keypoints1.clear();
    app.keypoints2.clear();
    app.descs1.clear();
    app.descs2.clear();
    app.kp1_colors.clear();
    app.kp2_colors.clear();
    app.ordered_keypoints1.clear();
    app.ordered_keypoints2.clear();
    app.matching_2_1.clear();
    app.matching_1_2.clear();

    app.pyramid_param.scale_factors.clear();
    app.pyramid_param.level_sigma2.clear();
    app.pyramid_param.inv_scale_factors.clear();
    app.pyramid_param.inv_level_sigma2.clear();
    app.pyramid_param.nb_feature_per_level.clear();
    app.pyramid_param.total_features = 0;
}

void app_prepare(App &app)
{
    init_pyramid_parameters(app.pyramid_param, 1, 1.2, 1000);

    if (app.method == STOCK_ORBSLAM)
    {
        orb_descriptors_orb_keypoints(app);
    }
    else if (app.method == TILDE_BRIEF)
    {
        brief_descriptors_tilde_keypoints(app);
    }
    else if (app.method == SURF_BRIEF)
    {
        brief_descriptors_surf_keypoints(app);
    }
    else if (app.method == SURF_ORB)
    {
        orb_descriptors_surf_keypoints(app);
    }

    app.matching_1_2.resize(app.keypoints1.size());
    get_inverted_matching(app.matching_1_2, app.matching_2_1);
}


