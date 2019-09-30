/**
* This file is part of ORB-SLAM2.
*
*/

#ifndef SURFEXTRACTOR_H
#define SURFEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

class SURFextractor
{
    public:
    SURFextractor(double hessianThreshold = 1000, int nOctaves = 4, int nOctaveLayers = 3);

    ~SURFextractor() {}

    void operator()(cv::InputArray             image,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::OutputArray            descriptors);

    protected:
    std::vector<cv::Point> pattern;

    cv::Ptr<cv::xfeatures2d::SURF> surf_detector;
};

#endif
