#ifndef SENS_WEBCAMERA_H
#define SENS_WEBCAMERA_H

#include <thread>
#include <atomic>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <sens/SENSCamera.h>

class SENSWebCamera : public SENSCameraBase
{
public:
    SENSWebCamera()
    {
        _permissionGranted = true;
    }

    const SENSCameraConfig& start(std::string                   deviceId,
                                  const SENSCameraStreamConfig& streamConfig,
                                  cv::Size                      imgBGRSize           = cv::Size(),
                                  bool                          mirrorV              = false,
                                  bool                          mirrorH              = false,
                                  bool                          convToGrayToImgManip = false,
                                  int                           imgManipWidth        = -1,
                                  bool                          provideIntrinsics    = true,
                                  float                         fovDegFallbackGuess  = 65.f) override;

    void stop() override;

    const SENSCaptureProperties& captureProperties() override;
    SENSFramePtr                 latestFrame() override;

    void grab();
    
private:
    cv::VideoCapture _videoCapture;
    
    std::atomic_bool _stop{false};
    std::thread _cameraThread;
    //current frame
    SENSFramePtr _sensFrame;
    std::mutex _frameMutex;
};

#endif //SENS_WEBCAMERA_H
