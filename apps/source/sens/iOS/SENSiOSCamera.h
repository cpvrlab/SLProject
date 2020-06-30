#include <SENSCamera.h>
#import "SENSiOSCameraDelegate.h"
#import <simd/matrix_types.h>

class SENSiOSCamera : public SENSCameraBase
{
public:
    SENSiOSCamera();
    ~SENSiOSCamera();

    //! on ios we can ignore fovDegFallbackGuess as the api provides dynamic camera intrinsics
    const SENSCameraConfig& start(std::string                   deviceId,
                                  const SENSCameraStreamConfig& streamConfig,
                                  cv::Size                      imgRGBSize           = cv::Size(),
                                  bool                          mirrorV              = false,
                                  bool                          mirrorH              = false,
                                  bool                          convToGrayToImgManip = false,
                                  cv::Size                      imgManipSize         = cv::Size(),
                                  bool                          provideIntrinsics    = true,
                                  float                         fovDegFallbackGuess  = 65.f) override;

    const SENSCameraConfig& start(SENSCameraFacing facing,
                                  float            approxHorizFov,
                                  cv::Size         imgRGBSize,
                                  bool             mirrorV              = false,
                                  bool             mirrorH              = false,
                                  bool             scaleImgRGB          = false,
                                  bool             convToGrayToImgManip = false,
                                  cv::Size         imgManipSize         = cv::Size(),
                                  bool             provideIntrinsics    = true,
                                  float            fovDegFallbackGuess  = 65.f) override;

    void                         stop() override;
    const SENSCaptureProperties& captureProperties() override;
    SENSFramePtr                 latestFrame() override;

private:
    void processNewFrame(unsigned char* data, int imgWidth, int imgHeight, matrix_float3x3* camMat3x3);

    SENSiOSCameraDelegate* _cameraDelegate;

    std::mutex   _processedFrameMutex;
    SENSFramePtr _processedFrame;
};
