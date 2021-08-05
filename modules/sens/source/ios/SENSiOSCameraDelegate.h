#ifndef SENS_IOSCAMERA_DELEGATE_H
#define SENS_IOSCAMERA_DELEGATE_H

#import <Foundation/Foundation.h>
#import <AVFoundation/AVCaptureOutput.h> // Allows us to use AVCaptureVideoDataOutputSampleBufferDelegate
#import <simd/matrix_types.h>

#include <vector>
#include <SENSCamera.h>

@interface SENSiOSCameraDelegate : NSObject<AVCaptureVideoDataOutputSampleBufferDelegate>

- (BOOL)startCamera:(NSString*)deviceId
                withWidth:(int)width
                andHeight:(int)height
           autoFocusState:(BOOL)autoFocusEnabled
  videoStabilizationState:(BOOL)videoStabilizationEnabled
          intrinsicsState:(BOOL)provideIntrinsics;
- (BOOL)stopCamera;

- (SENSCaptureProps)retrieveCaptureProperties;

@property (nonatomic, assign) std::function<void(unsigned char*, int, int, matrix_float3x3*)> updateCB;
@property (nonatomic, assign) std::function<void(bool)>                                       permissionCB;

@end

#endif
