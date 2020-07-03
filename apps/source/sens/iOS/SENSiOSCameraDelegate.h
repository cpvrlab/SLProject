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

- (SENSCaptureProperties)retrieveCaptureProperties;

@property (nonatomic, assign) std::function<void(unsigned char*, int, int, matrix_float3x3*)> callback;

@end
