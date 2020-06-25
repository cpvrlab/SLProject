#import <AVFoundation/AVCaptureSession.h>
#import <AVFoundation/AVCaptureDevice.h> // For access to the camera
#import <AVFoundation/AVCaptureInput.h>  // For adding a data input to the camera
#import <AVFoundation/AVCaptureOutput.h> // For capturing frames
#import <CoreVideo/CVPixelBuffer.h>      // for using pixel format types

#import "SENSiOSCameraDelegate.h"

#include <sens/SENSCamera.h>
#include <sens/SENSUtils.h>

@interface SENSiOSCameraDelegate () {
@private
    AVCaptureSession*         _captureSession; // Lets us set up and control the camera
    AVCaptureDevice*          _camera;         // A pointer to the front or to the back camera
    AVCaptureDeviceInput*     _cameraInput;    // This is the data input for the camera that allows us to capture frames
    AVCaptureVideoDataOutput* _videoOutput;    // For the video frame data from the camera
}

@end

@implementation SENSiOSCameraDelegate

- (id)init
{
    //Initialize the parent class(es) up the hierarchy and create self:
    self = [super init];

    //Initialize members (not necessary with ARC)
    _captureSession = nil;
    _camera         = nil;
    _cameraInput    = nil;
    _videoOutput    = nil;

    return self;
}

- (void)captureOutput:(AVCaptureOutput*)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection*)connection
{
    // Check if this is the output we are expecting:
    if (captureOutput == _videoOutput)
    {
        // If it's a video frame, copy and process it
        CVReturn         err;
        CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);

        CVPixelBufferLockBaseAddress(pixelBuffer, 0);

        int            imgWidth  = (int)CVPixelBufferGetWidth(pixelBuffer);
        int            imgHeight = (int)CVPixelBufferGetHeight(pixelBuffer);
        unsigned char* data      = (unsigned char*)CVPixelBufferGetBaseAddress(pixelBuffer);

        if (!data)
        {
            NSLog(@"No pixel buffer data");
            return;
        }

        if (_callback)
        {
            _callback(data, imgWidth, imgHeight);
        }

        CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    }
}

- (void)videoCameraStarted:(NSNotification*)note
{
    //Note: not used at the moment
    // This callback has done its job, now disconnect it
    [[NSNotificationCenter defaultCenter] removeObserver:self
                                                    name:AVCaptureSessionDidStartRunningNotification
                                                  object:_captureSession];
}

- (void)printActiveFormat
{
    if (_camera == nil)
        return;

    AVCaptureDeviceFormat* currFormat = [_camera activeFormat];
    CMFormatDescriptionRef formatDesc = [currFormat formatDescription];
    if (formatDesc)
    {
        CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(formatDesc);
        printf("active format: w %d h %d", dims.width, dims.height);
    }
}

- (BOOL)startCamera:(NSString*)deviceId withWidth:(int)width andHeight:(int)height
{
    // Make sure we initialize our camera pointer:
    _camera = nil;

    // specifying AVMediaTypeVideo will ensure we only get a list of cameras, no microphones
    NSArray* devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];

    for (AVCaptureDevice* device in devices)
    {
        if ([device uniqueID] == deviceId)
        {
            _camera = device;
            break;
        }
    }

    if (_camera == nil)
        return NO;

    //We set the format directly on the device to get all available configurations (not all are available on AVCaptureSession)
    //see: https://developer.apple.com/documentation/avfoundation/avcapturedevice?language=objc
    //see: https://stackoverflow.com/questions/36689578/avfoundation-capturing-video-with-custom-resolution/40097057
    //find best corresponding format
    AVCaptureDeviceFormat* bestFormat = nil;
    AVFrameRateRange*      bestRange  = nil;

    for (AVCaptureDeviceFormat* format in [_camera formats])
    {
        CMFormatDescriptionRef formatDesc = [format formatDescription];
        if (formatDesc)
        {
            CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(formatDesc);
            if (dims.width == width && dims.height == height)
            {
                for (AVFrameRateRange* range in [format videoSupportedFrameRateRanges])
                {
                    if (bestRange == nil || bestRange.maxFrameRate < range.maxFrameRate)
                    {
                        bestFormat = format;
                        bestRange  = range;
                    }
                }
            }
        }
    }

    // Set a frame rate for the camera:
    // We first need to lock the camera, so no one else can mess with its configuration:
    if ([_camera lockForConfiguration:nil])
    {
        // Set the device's active format.
        [_camera setActiveFormat:bestFormat];
        //[self printActiveFormat];

        // Set the device's min/max frame duration.
        CMTime duration = [bestRange minFrameDuration];
        [_camera setActiveVideoMinFrameDuration:duration];
        [_camera setActiveVideoMaxFrameDuration:duration];

        //Make sure we have a capture session
        if (nil == _captureSession)
        {
            _captureSession = [[AVCaptureSession alloc] init];
        }

        //disable that video device active format is overwritten
        [_captureSession setAutomaticallyConfiguresCaptureDeviceForWideColor:NO];
        //again: setting session present to AVCaptureSessionPresetInputPriority enabled that the previously set format on the video device is not overwritten
        if ([_captureSession canSetSessionPreset:AVCaptureSessionPresetInputPriority])
            [_captureSession setSessionPreset:AVCaptureSessionPresetInputPriority];

        // Plug camera and capture sesiossion together:
        {
            // Request a camera input from the camera
            NSError* error = nil;
            _cameraInput   = [AVCaptureDeviceInput deviceInputWithDevice:_camera
                                                                 error:&error];
            // Check if we've got any errors
            if (nil != error)
                return NO;

            // We've got the input from the camera, now attach it to the capture session:
            if ([_captureSession canAddInput:_cameraInput])
                [_captureSession addInput:_cameraInput];
            else
                return NO;
        }

        // Add the video output:
        {
            // Create the video data output
            _videoOutput = [[AVCaptureVideoDataOutput alloc] init];

            // Create a queue for capturing video frames
            dispatch_queue_t captureQueue = dispatch_queue_create("captureQueue", DISPATCH_QUEUE_SERIAL);

            // Use the AVCaptureVideoDataOutputSampleBufferDelegate capabilities of CameraDelegate
            [_videoOutput setSampleBufferDelegate:self queue:captureQueue];

            // Set up the video output:
            // Do we care about missing frames?
            [_videoOutput setAlwaysDiscardsLateVideoFrames:YES];

            // We want the frames in some RGB format
            NSNumber* framePixelFormat = [NSNumber numberWithInt:kCVPixelFormatType_32BGRA];
            _videoOutput.videoSettings = [NSDictionary dictionaryWithObject:framePixelFormat
                                                                     forKey:(id)kCVPixelBufferPixelFormatTypeKey];

            // Add the video data output to the capture session
            [_captureSession addOutput:_videoOutput];
        }

        // Set up a callback, so we are notified when the camera actually starts
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(videoCameraStarted:)
                                                     name:AVCaptureSessionDidStartRunningNotification
                                                   object:_captureSession];

        // Start the captureing
        [_captureSession startRunning];
        //[self printActiveFormat];

        [_camera unlockForConfiguration];
    }

    // Note: Returning true from this function only means that setting up went OK.
    // It doesn't mean that the camera has started yet.
    // We get notified about the camera having started in the videoCameraStarted() callback.
    return YES;
}

- (BOOL)stopCamera
{
    if (nil != _captureSession)
    {
        [_captureSession stopRunning];
        _captureSession = nil;
        _camera         = nil;
        _cameraInput    = nil;
        _videoOutput    = nil;
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
        std::string      deviceId = [[device uniqueID] UTF8String];
        SENSCameraFacing facing;
        //facing
        if (AVCaptureDevicePositionFront == [device position])
            facing = SENSCameraFacing::FRONT;
        else if (AVCaptureDevicePositionBack == [device position])
            facing = SENSCameraFacing::BACK;
        else
            facing = SENSCameraFacing::UNKNOWN;

        SENSCameraCharacteristics characs(deviceId, facing);

        NSArray<AVCaptureDeviceFormat*>* deviceFormats = [device formats];
        for (AVCaptureDeviceFormat* format in deviceFormats)
        {
            CMFormatDescriptionRef formatDesc = [format formatDescription];
            if (formatDesc)
            {
                CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(formatDesc);
                int               w    = dims.width;
                int               h    = dims.height;
                //printf("dims: w %d h %d\n", w, h);
                if (!characs.contains({w, h}))
                {
                    //calculate focal length in pixel from horizontal field of view
                    float horizFovDeg    = [format videoFieldOfView];
                    float focalLengthPix = SENS::calcFocalLengthPixFromFOVDeg(horizFovDeg, w);

                    characs.add(w, h, focalLengthPix);
                }
            }
        }

        characsVec.push_back(characs);
    }
    return characsVec;
}

@end
