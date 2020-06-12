#import <AVFoundation/AVCaptureSession.h>
#import <AVFoundation/AVCaptureDevice.h> // For access to the camera
#import <AVFoundation/AVCaptureInput.h> // For adding a data input to the camera
#import <AVFoundation/AVCaptureOutput.h> // For capturing frames
#import <CoreVideo/CVPixelBuffer.h> // for using pixel format types

#import "SENSiOSCameraDelegate.h"

// See Keeping your C native code reusable and independent of AIR
// at http://easynativeextensions.com/keeping-your-native-code-reusable/
//extern void sendMessage( const NSString * const messageType, const NSString * const message );

@interface SENSiOSCameraDelegate()
{
@private
    AVCaptureSession*         m_captureSession; // Lets us set up and control the camera
    AVCaptureDevice*          m_camera; // A pointer to the front or to the back camera
    AVCaptureDeviceInput*     m_cameraInput; // This is the data input for the camera that allows us to capture frames
    AVCaptureVideoDataOutput* m_videoOutput; // For the video frame data from the camera
}
@end
 
@implementation SENSiOSCameraDelegate
 
- (id)init
{
    // 1. Initialize the parent class(es) up the hierarchy and create self:
    self = [super init];
 
    // 2. Initialize members:
    m_captureSession    = NULL;
    m_camera            = NULL;
    m_cameraInput       = NULL;
    m_videoOutput       = NULL;
 
    return self;
}

- (BOOL)findCamera:(BOOL)useFrontCamera
{
    // 0. Make sure we initialize our camera pointer:
    m_camera = NULL;
 
    // 1. Get a list of available devices:
    // specifying AVMediaTypeVideo will ensure we only get a list of cameras, no microphones
    NSArray* devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
 
    // 2. Iterate through the device array and if a device is a camera, check if it's the one we want:
    for (AVCaptureDevice* device in devices)
    {
        NSString* uniqueId = [device uniqueID];
        if (useFrontCamera && AVCaptureDevicePositionFront == [device position])
        {
            // We asked for the front camera and got the front camera, now keep a pointer to it:
            m_camera = device;
        }
        else if (!useFrontCamera && AVCaptureDevicePositionBack == [device position])
        {
            // We asked for the back camera and here it is:
            m_camera = device;
        }
    }
 
    // 3. Set a frame rate for the camera:
    if (NULL != m_camera)
    {
        // We first need to lock the camera, so no one else can mess with its configuration:
        if ([m_camera lockForConfiguration:NULL])
        {
            // Set a minimum frame rate of 10 frames per second
            [m_camera setActiveVideoMinFrameDuration:CMTimeMake(1, 10)];
 
            // and a maximum of 30 frames per second
            [m_camera setActiveVideoMaxFrameDuration:CMTimeMake(1, 30)];
 
            [m_camera unlockForConfiguration];
        }
    }
 
    // 4. If we've found the camera we want, return true
    return (NULL != m_camera);
}

- (BOOL)attachCameraToCaptureSession
{
    // 0. Assume we've found the camera and set up the session first:
    assert(NULL != m_camera);
    assert(NULL != m_captureSession);
 
    // 1. Initialize the camera input
    m_cameraInput = NULL;
 
    // 2. Request a camera input from the camera
    NSError * error = NULL;
    m_cameraInput = [AVCaptureDeviceInput deviceInputWithDevice:m_camera
                                                          error:&error ];
 
    // 2.1. Check if we've got any errors
    if (NULL != error)
    {
        // TODO: send an error event to ActionScript
        return false;
    }
 
    // 3. We've got the input from the camera, now attach it to the capture session:
    if ([m_captureSession canAddInput:m_cameraInput])
    {
        [m_captureSession addInput:m_cameraInput];
    }
    else
    {
        // TODO: send an error event to ActionScript
        return false;
    }
 
    // 4. Done, the attaching was successful, return true to signal that
    return true;
}

- (void)setupVideoOutput
{
    // 1. Create the video data output
    m_videoOutput = [[AVCaptureVideoDataOutput alloc] init];
 
    // 2. Create a queue for capturing video frames
    dispatch_queue_t captureQueue = dispatch_queue_create("captureQueue", DISPATCH_QUEUE_SERIAL);
 
    // 3. Use the AVCaptureVideoDataOutputSampleBufferDelegate capabilities of CameraDelegate:
    [m_videoOutput setSampleBufferDelegate:self queue:captureQueue];
 
    // 4. Set up the video output
    // 4.1. Do we care about missing frames?
    [m_videoOutput setAlwaysDiscardsLateVideoFrames:YES];
 
    // 4.2. We want the frames in some RGB format, which is what ActionScript can deal with
    NSNumber* framePixelFormat  = [NSNumber numberWithInt:kCVPixelFormatType_32BGRA];
    m_videoOutput.videoSettings = [NSDictionary dictionaryWithObject:framePixelFormat
                                                              forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    
    // 5. Add the video data output to the capture session
    [m_captureSession addOutput:m_videoOutput];
}

- (void)captureOutput:(AVCaptureOutput*)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection*)connection
{
    // 1. Check if this is the output we are expecting:
    if (captureOutput == m_videoOutput)
    {
        // 2. If it's a video frame, copy it from the sample buffer:
        [self copyVideoFrame:sampleBuffer];
    }
}

- (void)copyVideoFrame:(CMSampleBufferRef)sampleBuffer
{
    CVReturn err;
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    CVPixelBufferLockBaseAddress(pixelBuffer,0);
    
    int imgWidth  = (int) CVPixelBufferGetWidth(pixelBuffer);
    int imgHeight = (int) CVPixelBufferGetHeight(pixelBuffer);
    unsigned char* data = (unsigned char*)CVPixelBufferGetBaseAddress(pixelBuffer);
        
    if(!data)
    {
        NSLog(@"No pixel buffer data");
        return;
    }
    
    cv::Mat rgba(imgHeight, imgWidth, CV_8UC4, (void*)data);
    cv::Mat rgbImg;
    cvtColor(rgba, rgbImg, cv::COLOR_RGBA2RGB, 3);
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

- (void)videoCameraStarted:(NSNotification*)note
{
    // This callback has done its job, now disconnect it
    [[NSNotificationCenter defaultCenter] removeObserver:self
                                                    name:AVCaptureSessionDidStartRunningNotification
                                                  object:m_captureSession];
 
    // Now send an event to ActionScript
    //sendMessage( @"CAMERA_STARTED_EVENT", @"" );
    //todo: how to log to ios console
}

- (BOOL)startCamera
{
    // 1. Find the back camera
    if ( ![self findCamera:false])
    {
        return false;
    }
 
    //2. Make sure we have a capture session
    if (NULL == m_captureSession)
    {
        m_captureSession = [[AVCaptureSession alloc] init];
    }
 
    // 3. Choose a preset for the session.
    // Optional TODO: You can parameterize this and set it in ActionScript.
    NSString* cameraResolutionPreset = AVCaptureSessionPreset640x480;
 
    // 4. Check if the preset is supported on the device by asking the capture session:
    if ( ![m_captureSession canSetSessionPreset:cameraResolutionPreset])
    {
        // Optional TODO: Send an error event to ActionScript
        return false;
    }
 
    // 4.1. The preset is OK, now set up the capture session to use it
    [m_captureSession setSessionPreset:cameraResolutionPreset];
 
    // 5. Plug camera and capture sesiossion together
    [self attachCameraToCaptureSession];
 
    // 6. Add the video output
    [self setupVideoOutput];
 
    // 7. Set up a callback, so we are notified when the camera actually starts
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(videoCameraStarted:)
                                                 name:AVCaptureSessionDidStartRunningNotification
                                               object:m_captureSession ];
 
    // 8. 3, 2, 1, 0... Start!
    [m_captureSession startRunning];
 
    // Note: Returning true from this function only means that setting up went OK.
    // It doesn't mean that the camera has started yet.
    // We get notified about the camera having started in the videoCameraStarted() callback.
    return true;
}

@end
