#ifndef SENS_WEBCAMERA_H
#define SENS_WEBCAMERA_H

#include <thread>
#include <atomic>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <SENSCamera.h>

class SENSWebCamera : public SENSBaseCamera
{
public:
    SENSWebCamera()
    {
        _permissionGranted = true;
    }

    const SENSCameraConfig& start(std::string                   deviceId,
                                  const SENSCameraStreamConfig& streamConfig,
                                  bool                          provideIntrinsics = true) override;

    void stop() override;

    const SENSCaptureProps& captureProperties() override;

    void grab();

private:
    cv::VideoCapture _videoCapture;

    std::atomic_bool _stop{false};
    std::thread      _cameraThread;
};

#endif //SENS_WEBCAMERA_H
