#import <Foundation/Foundation.h>
#import <AVFoundation/AVCaptureOutput.h> // Allows us to use AVCaptureVideoDataOutputSampleBufferDelegate
 
#include <vector>
#include <SENSCamera.h>

@interface SENSiOSCameraDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>

- (BOOL)startCamera:(NSString*)deviceId withWidth:(int)width andHeight:(int)height;
- (BOOL)stopCamera;

- (std::vector<SENSCameraCharacteristics>)getAllCameraCharacteristics;

@property (nonatomic, assign) std::function<void(unsigned char*, int, int)> callback;

@end
