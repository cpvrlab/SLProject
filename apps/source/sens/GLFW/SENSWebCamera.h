#ifndef SENS_WEBCAMERA_H
#define SENS_WEBCAMERA_H

#include <opencv2/opencv.hpp>
#include <SENSCamera.h>

class SENSWebCamera : public SENSCamera
{
public:
    SENSWebCamera(SENSCamera::Facing facing);
    ~SENSWebCamera();

    void         start(const Config config) override;
    void         start(int width, int height) override;
    void         stop(){};
    SENSFramePtr getLatestFrame() override;

private:
    cv::VideoCapture _videoCapture;
};

#endif //SENS_WEBCAMERA_H
