#ifndef SENS_IOSARCORE_DELEGATE_H
#define SENS_IOSARCORE_DELEGATE_H

//#import <UIKit/UIKit.h>
//#import <Metal/Metal.h>
//#import <MetalKit/MetalKit.h>
#import <ARKit/ARKit.h>
#import <simd/simd.h>

@interface SENSiOSARCoreDelegate : NSObject<ARSessionDelegate>

- (BOOL)isAvailable;

- (BOOL)start;
- (void)stop;

//pose, yPlane, uvPlane, width, height, intrinsic
@property (nonatomic, assign) std::function<void(simd_float4x4*, uint8_t*, uint8_t*, size_t, size_t, simd_float3x3*)> updateCB;
@property (nonatomic, assign) std::function<void(simd_float4x4* camPose, cv::Mat imgBGR, simd_float3x3* camMat3x3)> updateBgrCB;
@end

#endif
