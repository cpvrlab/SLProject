#ifndef BRIEF_DESCRIPTOR_H
#define BRIEF_DESCRIPTOR_H

#include "tools.h"

void ComputeBRIEFDescriptors(std::vector<Descriptor> &descriptors, const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,  const std::vector<cv::Point>* pattern = NULL);

void ComputeBRIEFDescriptors(std::vector<std::vector<Descriptor>> &descriptors, std::vector<cv::Mat> image_pyramid, PyramidParameters &p, std::vector<std::vector<cv::KeyPoint>>& allKeypoints);

#endif

