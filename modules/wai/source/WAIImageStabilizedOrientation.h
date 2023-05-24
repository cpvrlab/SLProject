#ifndef WAI_IMAGE_STABILIZED_ORIENTATION_H
#define WAI_IMAGE_STABILIZED_ORIENTATION_H

#include <opencv2/core/core.hpp>
#include <SLVec3.h>

class WAIImageStabilizedOrientation
{
public:
    bool findCameraOrientationDifferenceF2F(cv::Mat        imageGray, //for corner extraction
                                            cv::Mat&       imageRgb,
                                            const cv::Mat& intrinsic,
                                            float          scaleToGray,
                                            bool           decorate); //for debug decoration

    bool findCameraOrientationDifferenceF2FHorizon(const SLVec3f& horizon,
                                                   cv::Mat        imageGray, //for corner extraction
                                                   cv::Mat&       imageRgb,
                                                   const cv::Mat& intrinsic,
                                                   float          scaleToGray,
                                                   bool           decorate); //for debug decoration
private:
    cv::Mat _lastImageGray;
    cv::Mat _Tcw;
    float   _xAngRAD = 0.f;
    float   _yAngRAD = 0.f;
    float   _zAngRAD = 0.f;

    std::vector<cv::KeyPoint> _lastKeyPts;
    std::vector<cv::Point2f>  _lastPts;
    std::vector<cv::Point2f>  _currPts;
    std::vector<cv::Point2f>  _lastGoodPts;
    std::vector<cv::Point2f>  _currGoodPts;
    std::vector<uchar>        _inliers;
    std::vector<float>        _err;

    int _fIniThFAST = 30;
    int _fMinThFAST = 7;
};

#endif // WAI_IMAGE_STABILIZED_ORIENTATION_H
