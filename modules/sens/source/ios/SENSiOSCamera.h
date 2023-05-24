#ifndef SENS_IOSCAMERA_H
#define SENS_IOSCAMERA_H

#include <SENSCamera.h>
#import "SENSiOSCameraDelegate.h"
#import <simd/matrix_types.h>
#include <opencv2/opencv.hpp>

class SENSiOSCamera : public SENSBaseCamera
{
public:
    SENSiOSCamera();
    ~SENSiOSCamera();

    //! on ios we can ignore fovDegFallbackGuess as the api provides dynamic camera intrinsics
    const SENSCameraConfig& start(std::string                   deviceId,
                                  const SENSCameraStreamConfig& streamConfig,
                                  bool                          provideIntrinsics = true) override;

    void                    stop() override;
    const SENSCaptureProps& captureProperties() override;

private:
    void processNewFrame(unsigned char* data, int imgWidth, int imgHeight, matrix_float3x3* camMat3x3);
    void updatePermission(bool granted);

    SENSiOSCameraDelegate* _cameraDelegate;
};

#endif
