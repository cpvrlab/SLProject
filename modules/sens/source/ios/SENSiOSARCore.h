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

    bool init(unsigned int textureId = 0) override;
    bool isReady() override;
    bool resume() override;
    void reset() override;
    void pause() override;
    bool update(cv::Mat& pose) override;
    bool isAvailable() { return _available; };
    bool isInstalled() { return _available; };
    bool install() { return _available; };

    const SENSCameraConfig& start(std::string                   deviceId,
                                  const SENSCameraStreamConfig& streamConfig,
                                  bool                          provideIntrinsics) override;
    //from camera (does nothing in this case, because we have no image callbacks but we update arcore and the frame actively
    void stop() override { _started = false; };
    
    const SENSCaptureProperties& captureProperties() override;
    
private:
    void retrieveCaptureProperties();
    bool _available = false;
    SENSiOSARCoreDelegate* _arcoreDelegate;
};

#endif
