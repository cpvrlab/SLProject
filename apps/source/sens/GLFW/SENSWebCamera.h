#ifndef SENS_WEBCAMERA_H
#define SENS_WEBCAMERA_H

#include <opencv2/opencv.hpp>
#include <SENSCamera.h>

class SENSWebCamera : public SENSCamera
{
public:
    ~SENSWebCamera();

    void         init(SENSCamera::Facing facing) override;
    void         start(const Config config) override;
    void         start(int width, int height) override;
    void         stop() { _started = false; }
    SENSFramePtr getLatestFrame() override;

private:
    cv::VideoCapture      _videoCapture;
    std::vector<cv::Size> _streamSizes;
};

#endif //SENS_WEBCAMERA_H
