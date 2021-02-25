#ifndef SENS_IOSARCORE_H
#define SENS_IOSARCORE_H

#include <SENSARCore.h>

#import "SENSiOSARCoreDelegate.h"
#import <simd/simd.h>

class SENSiOSARCore : public SENSARCore
{
public:
    SENSiOSARCore();
    ~SENSiOSARCore()
    {
    }

    bool init() override;
    bool isReady() override;
    bool resume() override;
    void reset() override;
    void pause() override;
    bool update(cv::Mat& pose) override;

    const SENSCameraConfig& start(std::string                   deviceId,
                                  const SENSCameraStreamConfig& streamConfig,
                                  bool                          provideIntrinsics) override;
    //from camera (does nothing in this case, because we have no image callbacks but we update arcore and the frame actively
    void stop() override { _started = false; };
    
    const SENSCaptureProperties& captureProperties() override;
    
private:
    void retrieveCaptureProperties();
    
    SENSiOSARCoreDelegate* _arcoreDelegate;
};

#endif
