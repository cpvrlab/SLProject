#include <WAICompassAlignment.h>
#include <opencv2/imgproc.hpp>
#include <Profiler.h>

#include <iostream>

void WAICompassAlignment::setTemplate(cv::Mat& templateImage, double latitudeDEG, double longitudeDEG, double altitudeM)
{
    _template.image        = templateImage.clone();
    _template.latitudeDEG  = latitudeDEG;
    _template.longitudeDEG = longitudeDEG;
    _template.altitudeM    = altitudeM;
}

void WAICompassAlignment::update(const cv::Mat& frameGray,
                                 cv::Mat&       resultImage,
                                 const float    hFov,
                                 double         latitudeDEG,
                                 double         longitudeDEG,
                                 double         altitudeM,
                                 cv::Point&     vecCurForward)
{
    PROFILE_FUNCTION();

    // TODO(dgj1): handle error if no template has been set

    cv::matchTemplate(frameGray, _template.image, resultImage, cv::TM_CCOEFF_NORMED);

    double    minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(resultImage, &minVal, &maxVal, &minLoc, &maxLoc);

    std::cout << "maxLoc: " << maxLoc << std::endl;

    cv::Point imCenter       = cv::Point((int)(frameGray.cols * 0.5f),
                                   (int)(frameGray.rows * 0.5f));
    cv::Point tplCenter      = cv::Point((int)(_template.image.cols * 0.5f),
                                    (int)(_template.image.rows * 0.5f));
    cv::Point tplMatchCenter = maxLoc + tplCenter;

    cv::Point offset = tplMatchCenter - imCenter;

    float degPerPixelH = hFov / frameGray.cols;

    // Angle between camera forward and template match center
    // If GPS is absolutely precise, the difference between this angle and
    // the angle between GPS forward and template match center should be 0
    float angleCenterTemplateDegH = offset.x * degPerPixelH;

    // TODO(dgj1): actually calculate this
    cv::Point vecCurTpl = cv::Point((int)(_template.latitudeDEG - latitudeDEG),
                                    (int)(_template.longitudeDEG - longitudeDEG));
    vecCurTpl /= cv::norm(vecCurTpl);
    vecCurForward /= cv::norm(vecCurForward);
    float angleGPSTemplateDegH = (float)vecCurTpl.dot(vecCurForward);

    _rotAngDEG = angleGPSTemplateDegH - angleCenterTemplateDegH;
}
