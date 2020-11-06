#include <WAICompassAlignment.h>
#include <opencv2/imgproc.hpp>
#include <Instrumentor.h>

WAICompassAlignment::WAICompassAlignment(cv::Mat& templateImage)
{
    _templateImage = templateImage.clone();
}

void WAICompassAlignment::update(const cv::Mat& frameGray, cv::Mat& resultImage)
{
    PROFILE_FUNCTION();

    cv::matchTemplate(frameGray, _templateImage, resultImage, cv::TM_CCOEFF_NORMED);
}
