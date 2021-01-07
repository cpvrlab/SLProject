#ifndef SENS_UTILS_H
#define SENS_UTILS_H

#include <opencv2/opencv.hpp>

#define SENS_PI 3.1415926535897932384626433832795
#define SENS_DEG2RAD SENS_PI / 180.0
#define SENS_RAD2DEG 180.0 / SENS_PI

namespace SENS
{
//returns true if crop was calculated
bool calcCrop(cv::Size inputSize, float targetWdivH, int& cropW, int& cropH, int& width, int& height);
void cropImage(cv::Mat& img, float targetWdivH, int& cropW, int& cropH);
void cropImageTo(const cv::Mat& img, cv::Mat& outImg, float targetWdivH, int& cropW, int& cropH);

void mirrorImage(cv::Mat& img, bool mirrorH, bool mirrorV);
void extendWithBars(cv::Mat& img, float targetWdivH);

//calculate field of view in degree from focal length in pixel. (If you want to know the horizontal field of view you have to pass the image width as imgLength, for vertical field of view pass the image height
float calcFOVDegFromFocalLengthPix(const float focalLengthPix, const int imgLength);
//calculate focal length in pix from field of view in degree. (If you pass the horizontal field of view you have to pass the image width as imgLength. If you pass the vertical field of view you have to pass the image height as imgLength.
float calcFocalLengthPixFromFOVDeg(const float fovDeg, const int imgLength);

/*
 @brief calculate an unknown field of view (e.g. vertical) from a known field of view (e.g. horizontal)
 @param otherFovDeg specifies the known field of view in degree
 @param otherLength specifies the length belonging to the known field of view (e.g. width belongs to horizontal fovV and height belongs to vertical fov)
 @param length specifies the relative length (repecting the img aspect ration) in the direction of the unknown field of view
 @returns field of view in degree
*/
float calcFovDegFromOtherFovDeg(const float otherFovDeg, const int otherLength, const int length);
/*
 @brief calculate a scaled camera matrix using new and old reference lengths
 @param origMat specifies the original camera matrix
 @param newRefLength specifies the new reference length
 @param oldRefLength specifies the old reference length
 @returns adapted camera matrix
*/
cv::Mat adaptCameraMat(cv::Mat origMat, int newRefLength, int oldRefLength);

/*
 @brief calculate increased fov: e.g. we know the fov of a, image (oldFovDeg) with height heightOld that is vertically centered in a screen with heightNew and want to know the corresponding new field of view
 @param oldFovDeg old field of view in degree
 @param oldImgLength specifies corresponding old img length (e.g. combine vertical fov with height and horizontal fov with width)
 @param newImgLength specifies corresponding new img length (e.g. combine vertical fov with height and horizontal fovV with width)
 @returns new field of view in degree
 */
//todo: does not give correct results, I wonder why...
//float calcFovOfCenteredImg(const float oldFovDeg, const int oldImgLength, const int newImgLength);

};

#endif //SENS_UTILS_H
