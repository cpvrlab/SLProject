#ifndef SENS_IOSARKIT_DELEGATE_H
#define SENS_IOSARKIT_DELEGATE_H

#import <ARKit/ARKit.h>
#import <simd/simd.h>
#include <SENSCamera.h>

@interface SENSiOSARCoreDelegate : NSObject<ARSessionDelegate>

- (BOOL)isAvailable;

- (BOOL)run;
- (void)pause;
- (BOOL)reset;

- (void)latestFrame:(cv::Mat*)pose
            withImg:(cv::Mat*)imgBGR
       AndIntrinsic:(cv::Mat*)intrinsic
        AndImgWidth:(int*)w
       AndImgHeight:(int*)h
         IsTracking:(BOOL*)isTracking
     WithPointClout:(cv::Mat*)pc;
- (void)initARKit;

//pose, yPlane, uvPlane, width, height, intrinsic
@property (nonatomic, assign) std::function<void(simd_float4x4*, uint8_t*, uint8_t*, size_t, size_t, simd_float3x3*, bool isTracking)> updateCB;
@property (nonatomic, assign) std::function<void(simd_float4x4* camPose, cv::Mat imgBGR, simd_float3x3* camMat3x3)>                    updateBgrCB;
@end

#endif
