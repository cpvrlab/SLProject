#import <AVFoundation/AVCaptureSession.h>
#import <AVFoundation/AVCaptureDevice.h> // For access to the camera
#import <AVFoundation/AVCaptureInput.h> // For adding a data input to the camera
#import <AVFoundation/AVCaptureOutput.h> // For capturing frames
#import <CoreVideo/CVPixelBuffer.h> // for using pixel format types

#import "SENSiOSCameraDelegate.h"

#include <sens/SENSCamera.h>
#include <sens/SENSUtils.h>


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
 
    //Initialize members (not necessary with ARC)
    m_captureSession    = nil;
    m_camera            = nil;
    m_cameraInput       = nil;
    m_videoOutput       = nil;

    return self;
}

- (BOOL)attachCameraToCaptureSession
{
    // Assume we've found the camera and set up the session first:
    assert(nil != m_camera);
    assert(nil != m_captureSession);
 
    // Initialize the camera input
    m_cameraInput = nil;
 
    // Request a camera input from the camera
    NSError* error = nil;
    m_cameraInput = [AVCaptureDeviceInput deviceInputWithDevice:m_camera
                                                          error:&error ];
 
    // Check if we've got any errors
    if (nil != error)
        return false;
 
    // We've got the input from the camera, now attach it to the capture session:
    if ([m_captureSession canAddInput:m_cameraInput])
        [m_captureSession addInput:m_cameraInput];
    else
        return false;
 
    return true;
}

- (void)setupVideoOutput
{
    // Create the video data output
    m_videoOutput = [[AVCaptureVideoDataOutput alloc] init];
 
    // Create a queue for capturing video frames
    dispatch_queue_t captureQueue = dispatch_queue_create("captureQueue", DISPATCH_QUEUE_SERIAL);
 
    // Use the AVCaptureVideoDataOutputSampleBufferDelegate capabilities of CameraDelegate
    [m_videoOutput setSampleBufferDelegate:self queue:captureQueue];
 
    // Set up the video output:
    // Do we care about missing frames?
    [m_videoOutput setAlwaysDiscardsLateVideoFrames:YES];
 
    // We want the frames in some RGB format
    NSNumber* framePixelFormat  = [NSNumber numberWithInt:kCVPixelFormatType_32BGRA];
    m_videoOutput.videoSettings = [NSDictionary dictionaryWithObject:framePixelFormat
                                                              forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    
    // Add the video data output to the capture session
    [m_captureSession addOutput:m_videoOutput];
}

- (void)captureOutput:(AVCaptureOutput*)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection*)connection
{
    // Check if this is the output we are expecting:
    if (captureOutput == m_videoOutput)
    {
        // If it's a video frame, copy and process it
        [self processFrame:sampleBuffer];
    }
}

- (void)processFrame:(CMSampleBufferRef)sampleBuffer
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
    
    if(_callback)
    {
        _callback(data, imgWidth, imgHeight);
    }

    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

- (void)videoCameraStarted:(NSNotification*)note
{
    // This callback has done its job, now disconnect it
    [[NSNotificationCenter defaultCenter] removeObserver:self
                                                    name:AVCaptureSessionDidStartRunningNotification
                                                  object:m_captureSession];
}

+ (NSString*)getCaptureSessionPresentWithWidth:(int)width andHeight:(int)height
{
    return [NSString stringWithFormat:@"AVCaptureSessionPreset%dx%d",width, height];
}

- (void)printActiveFormat
{
    if(m_camera == nil)
        return;
    
    AVCaptureDeviceFormat* currFormat = [m_camera activeFormat];
    CMFormatDescriptionRef formatDesc = [currFormat formatDescription];
    if(formatDesc)
    {
        CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(formatDesc);
        printf("active format: w %d h %d", dims.width, dims.height);
    }
}

- (BOOL)startCamera:(NSString*)deviceId withWidth:(int)width andHeight:(int)height
{
    // Make sure we initialize our camera pointer:
    m_camera = nil;
  
    // specifying AVMediaTypeVideo will ensure we only get a list of cameras, no microphones
    NSArray* devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];

    for (AVCaptureDevice* device in devices)
    {
        if([device uniqueID] == deviceId)
        {
            m_camera = device;
            break;
        }
    }
    
    if(m_camera == nil)
        return NO;
    
    //We set the format directly on the device to get all available configurations (not all are available on AVCaptureSession)
    //see: https://developer.apple.com/documentation/avfoundation/avcapturedevice?language=objc
    //see: https://stackoverflow.com/questions/36689578/avfoundation-capturing-video-with-custom-resolution/40097057
    //find best corresponding format
    AVCaptureDeviceFormat* bestFormat = nil;
    AVFrameRateRange* bestRange = nil;
    for(AVCaptureDeviceFormat* format in [m_camera formats])
    {
        CMFormatDescriptionRef formatDesc = [format formatDescription];
        if(formatDesc)
        {
            CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(formatDesc);
            if( dims.width == width && dims.height == height )
            {
                for( AVFrameRateRange* range in [format videoSupportedFrameRateRanges])
                {
                    if(bestRange == nil || bestRange.maxFrameRate < range.maxFrameRate)
                    {
                        bestFormat = format;
                        bestRange = range;
                    }
                }
            }
        }
    }

    // Set a frame rate for the camera:
    // We first need to lock the camera, so no one else can mess with its configuration:
    if ([m_camera lockForConfiguration:nil])
    {
        // Set the device's active format.
        [m_camera setActiveFormat:bestFormat];
        //AVCaptureDeviceFormat* currFormat = [m_camera activeFormat];
        [self printActiveFormat];


        // Set the device's min/max frame duration.
        CMTime duration = [bestRange minFrameDuration];
        [m_camera setActiveVideoMinFrameDuration:duration];
        [m_camera setActiveVideoMaxFrameDuration:duration];
        


        //Make sure we have a capture session
        if (nil == m_captureSession)
        {
            m_captureSession = [[AVCaptureSession alloc] init];
        }
        [m_captureSession setAutomaticallyConfiguresCaptureDeviceForWideColor:NO];
        if ([m_captureSession canSetSessionPreset:AVCaptureSessionPresetInputPriority])
            [m_captureSession setSessionPreset:AVCaptureSessionPresetInputPriority];
        
        [self printActiveFormat];
        // Definition of AVCaptureSessionPreset string (e.g. AVCaptureSessionPreset3840x2160)
        //NSString* cameraResolutionPreset = [SENSiOSCameraDelegate getCaptureSessionPresentWithWidth: width andHeight:height];
        // Check if the preset is supported on the device by asking the capture session:
        //if ( ![m_captureSession canSetSessionPreset:cameraResolutionPreset])
        //    return false;
     
        // The preset is OK, now set up the capture session to use it
        //cameraResolutionPreset = @"AVCaptureSessionPreset2048x1536";
        //[m_captureSession setSessionPreset:cameraResolutionPreset];

        //[m_captureSession setSessionPreset:AVCaptureSessionPresetPhoto];
        
        // Plug camera and capture sesiossion together
        [self attachCameraToCaptureSession];
     
        // Add the video output
        [self setupVideoOutput];
     
        // Set up a callback, so we are notified when the camera actually starts
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(videoCameraStarted:)
                                                     name:AVCaptureSessionDidStartRunningNotification
                                                   object:m_captureSession ];
     
        // Start the captureing
        [m_captureSession startRunning];
        [self printActiveFormat];
           // [m_camera unlockForConfiguration];
    }
 
    // Note: Returning true from this function only means that setting up went OK.
    // It doesn't mean that the camera has started yet.
    // We get notified about the camera having started in the videoCameraStarted() callback.
    return true;
}

- (BOOL)stopCamera
{
    if (nil != m_captureSession)
    {
        [m_captureSession stopRunning];
        m_captureSession = nil;
        m_camera = nil;
        m_cameraInput = nil;
        m_videoOutput = nil;
    }
    return true;
}

- (SENSCaptureProperties)retrieveCaptureProperties
{
    SENSCaptureProperties characsVec;
    
    // specifying AVMediaTypeVideo will ensure we only get a list of cameras, no microphones
    NSArray* devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];

    for (AVCaptureDevice* device in devices)
    {
        //device id
        std::string deviceId = [[device uniqueID] UTF8String];
        SENSCameraFacing facing;
        //facing
        if (AVCaptureDevicePositionFront == [device position])
            facing = SENSCameraFacing::FRONT;
        else if (AVCaptureDevicePositionBack == [device position])
            facing = SENSCameraFacing::BACK;
        else
            facing = SENSCameraFacing::UNKNOWN;
        
        SENSCameraCharacteristics characs(deviceId, facing);
        //Make sure we have a capture session to retrieve AVCaptureSessionPresent
        //if (nil == m_captureSession)
        //{
        //    m_captureSession = [[AVCaptureSession alloc] init];
        //}
                
        NSArray<AVCaptureDeviceFormat*>* deviceFormats = [device formats];
        /*
        for (AVCaptureDeviceFormat* format in deviceFormats)
        {
            CMVideoDimensions resolution = format.highResolutionStillImageDimensions;
            int w = resolution.width;
            int h = resolution.height;
            printf("high dims: w %d h %d\n", w, h);
        }
         */
        /*
        AVCaptureDeviceFormat* bestFormat = nil;
        AVFrameRateRange* bestRange = nil;
        for(AVCaptureDeviceFormat* format in deviceFormats)
        {
            for( AVFrameRateRange* range in [format videoSupportedFrameRateRanges])
            {
                if(bestRange == nil || bestRange.maxFrameRate > range.maxFrameRate)
                {
                    bestFormat = format;
                    bestRange = range;
                }
            }
        }
         */
        
        for(AVCaptureDeviceFormat* format in deviceFormats)
        {
            CMFormatDescriptionRef formatDesc = [format formatDescription];
            if(formatDesc)
            {
                CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(formatDesc);
                int w = dims.width;
                int h = dims.height;
                printf("dims: w %d h %d\n", w, h);
                //NSString* cameraResolutionPreset = [SENSiOSCameraDelegate getCaptureSessionPresentWithWidth:w andHeight:h];
                //if ([m_captureSession canSetSessionPreset:cameraResolutionPreset])
                {
                    //float minFrameRate = [format
                    if(!characs.contains({w, h}))
                    {
                        //calculate focal length in pixel from horizontal field of view
                        float horizFovDeg = [format videoFieldOfView];
                        float focalLengthPix = SENS::calcFocalLengthPixFromFOVDeg(horizFovDeg, w);
                        
                        characs.add(w, h, focalLengthPix);
                    }
                }
            }
        }

        characsVec.push_back(characs);
    }
    return characsVec;
}

@end
