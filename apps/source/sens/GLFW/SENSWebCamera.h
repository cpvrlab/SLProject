#ifndef SENS_WEBCAMERA_H
#define SENS_WEBCAMERA_H

#include <opencv2/opencv.hpp>
#include <SENSCamera.h>
#include <thread>

class SENSWebCamera : public SENSCamera
{
public:
    SENSWebCamera()
    {
        _permissionGranted = true;
    }
    ~SENSWebCamera();

    void init(SENSCamera::Facing facing) override;
    void start(const Config config) override;
    void start(int width, int height) override;
    void stop() override;

    SENSFramePtr getLatestFrame() override;

private:
    void openCamera();

    bool                  _isStarting = false;
    cv::VideoCapture      _videoCapture;
    std::vector<cv::Size> _streamSizes;

    std::thread _thread;
};

#endif //SENS_WEBCAMERA_H
