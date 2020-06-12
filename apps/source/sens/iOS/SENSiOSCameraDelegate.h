#import <Foundation/Foundation.h>
#import <AVFoundation/AVCaptureOutput.h> // Allows us to use AVCaptureVideoDataOutputSampleBufferDelegate
 
@interface SENSiOSCameraDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>

//- ( id ) init;
- (BOOL)startCamera;

@end
