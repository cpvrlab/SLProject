#ifndef WAI_COMPASS_ALIGNMENT_H
#define WAI_COMPASS_ALIGNMENT_H

#include <opencv2/core.hpp>

class WAICompassAlignment
{
    struct Template
    {
        cv::Mat image;
        double  latitudeDEG;
        double  longitudeDEG;
        double  altitudeM;
    };

public:
    // TODO(dgj1): overload for function without getting the resultImage
    void update(const cv::Mat& frameGray,
                cv::Mat&       resultImage,
                const float    hFov,
                double         latitudeDEG,
                double         longitudeDEG,
                double         altitudeM,
                cv::Point&     forwardPoint);

    void  setTemplate(cv::Mat& templateImage, double latitudeDEG, double longitudeDEG, double altitudeM);
    float getRotAngleDEG() { return _rotAngDEG; }

private:
    // TODO(dgj1): allow multiple templates
    Template _template;
    float    _rotAngDEG = 0.0f;
};

#endif