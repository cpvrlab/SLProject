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

    bool init(int targetWidth, int targetHeight, int manipWidth, int manipHeight, bool convertManipToGray) override;
    bool isReady() override;
    bool resume() override;
    void reset() override;
    void pause() override;
    bool update(cv::Mat& pose) override;

private:
    SENSiOSARCoreDelegate* _arcoreDelegate;
};

#endif
