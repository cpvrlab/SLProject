#ifndef SENS_WEBCAMERA_H
#define SENS_WEBCAMERA_H

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
                                  cv::Size                      imgRGBSize           = cv::Size(),
                                  bool                          mirrorV              = false,
                                  bool                          mirrorH              = false,
                                  bool                          convToGrayToImgManip = false,
                                  cv::Size                      imgManipSize         = cv::Size(),
                                  bool                          provideIntrinsics    = true,
                                  float                         fovDegFallbackGuess  = 65.f) override;

    void stop() override;

    const SENSCaptureProperties& captureProperties() override;
    SENSFramePtr                 latestFrame() override;

private:
    cv::VideoCapture _videoCapture;
};

#endif //SENS_WEBCAMERA_H
