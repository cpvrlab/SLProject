#ifndef SENS_UTILS_H
#define SENS_UTILS_H

#include <opencv2/opencv.hpp>

namespace SENS
{

void cropImage(cv::Mat& img, float targetWdivH, int& cropW, int& cropH);
void mirrorImage(cv::Mat& img, bool mirrorH, bool mirrorV);
};

#endif //SENS_UTILS_H
