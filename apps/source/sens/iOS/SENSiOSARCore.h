#ifndef SENS_IOSARCORE_H
#define SENS_IOSARCORE_H

#include <sens/SENSARCore.h>
#import "SENSiOSARCoreDelegate.h"
#import <simd/simd.h>

class SENSiOSARCore : public SENSARCore
{
public:
    SENSiOSARCore();
    
    bool init(int targetWidth, int targetHeight, int manipWidth, int manipHeight, bool convertManipToGray) override;
    bool isReady() override;
    bool resume() override;
    void reset() override;
    void pause() override;
    bool update(cv::Mat& intrinsic, cv::Mat& view) override;
    SENSFramePtr latestFrame() override;
    void setDisplaySize(int w, int h) override;

private:
    void onUpdate(simd_float4x4* camPose, unsigned char* data, int imgWidth, int imgHeight, simd_float3x3* camMat3x3);
    
    SENSiOSARCoreDelegate* _arcoreDelegate;
/*
    bool start() override;
    void stop() override;

private:
    //callback from delegate
    void updateLocation(double latitudeDEG,
                        double longitudeDEG,
                        double altitudeM,
                        double accuracyM);
    //callback for permission update
    void updatePermission(bool granted);
    
    SENSiOSGpsDelegate* _gpsDelegate;
 */
};

#endif
