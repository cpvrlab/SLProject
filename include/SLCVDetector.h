#ifndef SLCVDETECTOR_H
#define SLCVDETECTOR_H
#include <SLCV.h>
#include <opencv2/xfeatures2d.hpp>

class SLCVDetector
{
private:
    cv::Ptr<cv::FeatureDetector> _detector;
public:
    SLCVDetector(SLCVDetectorType type, SLbool force=false);
    SLbool forced;
    SLCVDetectorType type;
    void detect(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::InputArray mask = cv::noArray());

    void setDetector(cv::Ptr<cv::FeatureDetector> detector) { _detector = detector; }
};

#endif // SLCVDETECTOR_H
