#ifndef SENS_UTILS_H
#define SENS_UTILS_H

#include <opencv2/opencv.hpp>

#define SENS_PI 3.1415926535897932384626433832795
#define SENS_DEG2RAD SENS_PI / 180.0
#define SENS_RAD2DEG 180.0 / SENS_PI

namespace SENS
{
void cropImage(cv::Mat& img, float targetWdivH, int& cropW, int& cropH);
void mirrorImage(cv::Mat& img, bool mirrorH, bool mirrorV);
void extendWithBars(cv::Mat& img, float targetWdivH, int cvBorderType, int& addW, int& addH);

//calculate field of view in degree from focal length in pixel. (If you want to know the horizontal field of view you have to pass the image width as imgLength, for vertical field of view pass the image height
float calcFOVDegFromFocalLengthPix(const float focalLengthPix, const int imgLength );
//calculate focal length in pix from field of view in degree. (If you pass the horizontal field of view you have to pass the image width as imgLength. If you pass the vertical field of view you have to pass the image height as imgLength.
float calcFocalLengthPixFromFOVDeg(const float fovDeg, const int imgLength);
};

#endif //SENS_UTILS_H
