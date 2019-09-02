#ifndef MATCHING_H
#define MATCHING_H

#include "tools.h"


void match_keypoints_0(std::vector<int> &indexes,
                       std::vector<cv::KeyPoint> &kps1, std::vector<Descriptor> &desc1, 
                       std::vector<cv::KeyPoint> &kps2, std::vector<Descriptor> &desc2,
                       float thres = 30);


void match_keypoints_1(std::vector<int> &indexes,
                       std::vector<cv::KeyPoint> &kps1, std::vector<Descriptor> &desc1, 
                       std::vector<cv::KeyPoint> &kps2, std::vector<Descriptor> &desc2,
                       bool check_orientation = false,
                       //float factor = 0.08333333,
                       float factor = 0.03333333,
                       float nnratio = 0.70,
                       float thres = 70);

#endif

