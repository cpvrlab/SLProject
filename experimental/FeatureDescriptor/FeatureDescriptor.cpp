#include <iostream>
#include <stdio.h>
#include "tools.h"

int main(int argc, char** argv)
{
    cv::Mat img1_resized;
    cv::Mat img2_resized;

    cv::Mat img1;
    cv::Mat img2;

    cv::Mat img_gray1;
    cv::Mat img_gray2;

    Descriptor desc1;
    Descriptor desc2;

    std::vector<int> umax;

    if (argc < 3)
    {
        std::cout << "usage:" << std::endl << argv[0] << " image1 image2" << std::endl;
        exit(1);
    }

    img1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    img2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

    cv::resize(img1, img1_resized, cv::Size(512,512),0,0,cv::INTER_NEAREST);
    cv::resize(img2, img2_resized, cv::Size(512,512),0,0,cv::INTER_NEAREST);

    img_gray1 = to_grayscale(img1);
    img_gray2 = to_grayscale(img2);

    init_patch(umax);

    cv::GaussianBlur(img_gray1, img_gray1, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
    cv::GaussianBlur(img_gray2, img_gray2, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

    cv::KeyPoint kp1 = get_middle_keypoint(img_gray1);
    cv::KeyPoint kp2 = get_middle_keypoint(img_gray2);

    keypoint_angle(img_gray1, kp1, umax);
    keypoint_angle(img_gray2, kp2, umax);





    std::cout << "angle img1 : " << kp1.angle * 180.0/M_PI << std::endl;
    std::cout << "angle img2 : " << kp2.angle * 180.0/M_PI << std::endl;
    std::cout << "d angle : " << (kp1.angle - kp2.angle) * 180.0/M_PI  << std::endl;

    cv::imshow("img1", img1_resized);
    cv::imshow("img2", img2_resized);
    cv::waitKey(0);

    return 0;
}


