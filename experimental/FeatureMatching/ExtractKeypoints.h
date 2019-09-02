#ifndef EXTRACTKEYPOINTS_H
#define EXTRACTKEYPOINTS_H

#include "tools.h"

void KPExtractOrbSlam(std::vector<std::vector<cv::KeyPoint>>& allKeypoints, std::vector<cv::Mat> &image_pyramid, PyramidParameters &p, float iniThFAST = 20, float minThFAST = 7);

void KPExtractTILDE(std::vector<cv::KeyPoint>&allKeypoints, cv::Mat image);

void KPExtractSURF(std::vector<cv::KeyPoint>& allKeypoints, cv::Mat image);

#endif

