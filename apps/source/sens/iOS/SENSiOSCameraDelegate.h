#import <Foundation/Foundation.h>
#import <AVFoundation/AVCaptureOutput.h> // Allows us to use AVCaptureVideoDataOutputSampleBufferDelegate

#include <vector>
#include <SENSCamera.h>

@interface SENSiOSCameraDelegate : NSObject<AVCaptureVideoDataOutputSampleBufferDelegate>

- (BOOL)startCamera:(NSString*)deviceId
                withWidth:(int)width
                andHeight:(int)height
           autoFocusState:(BOOL)autoFocusEnabled
  videoStabilizationState:(BOOL)videoStabilizationEnabled;
- (BOOL)stopCamera;

- (SENSCaptureProperties)retrieveCaptureProperties;

@property (nonatomic, assign) std::function<void(unsigned char*, int, int)> callback;

@end
