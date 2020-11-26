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

@property (nonatomic, assign) std::function<void(simd_float4x4*, unsigned char*, int, int, simd_float3x3*)> updateCB;

@end

#endif
