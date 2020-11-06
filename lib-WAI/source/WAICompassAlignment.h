#ifndef WAI_COMPASS_ALIGNMENT_H
#define WAI_COMPASS_ALIGNMENT_H

#include <opencv2/core.hpp>

class WAICompassAlignment
{
public:
    WAICompassAlignment(cv::Mat& templateImage);
    // TODO(dgj1): overload for function without getting the resultImage
    void update(const cv::Mat& frameGray, cv::Mat& resultImage);

private:
    cv::Mat _templateImage;
};

#endif