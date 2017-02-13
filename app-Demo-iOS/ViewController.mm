//
//  ViewController.m
//  comgr
//
//  Created by Marcus Hudritsch on 30.11.11.
//  Copyright (c) 2011 __MyCompanyName__. All rights reserved.
//

#import "ViewController.h"
#import <CoreMotion/CoreMotion.h>

// The only C-interface to include for the SceneLibrary

#include <SLInterface.h>
#include <SLImage.h>
#include <SLCVCapture.h>
#include <mach/mach_time.h>

//-----------------------------------------------------------------------------
// C-Prototypes
float GetSeconds();
SLbool onPaintRTGL();
//-----------------------------------------------------------------------------
// Global pointer to the GLView instance that can be accessed by onPaintRTGL
GLKView* myView = 0;
//-----------------------------------------------------------------------------
// Global SLSceneView handle
int svIndex = 0;
//-----------------------------------------------------------------------------
// Global screen scale (2.0 for retina, 1.0 else)
float screenScale = 1.0f;

//-----------------------------------------------------------------------------
// C-Function used as C-function callback for raytracing update
SLbool onPaintRTGL()
{
   [myView display];
   return true;
}
//-----------------------------------------------------------------------------
/*!
 Returns the absolute time in seconds since the system started. It is based
 on a CPU clock counter.
 */
float GetSeconds()
{
    static mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    uint64_t now = mach_absolute_time();
    now *= info.numer;
    now /= info.denom;
    double sec = (double)now / 1000000000.0;
    return (float)sec;
}
//-----------------------------------------------------------------------------
@interface ViewController ()
{
    SLfloat  m_lastFrameTimeSec;  //!< Timestamp for passing highres time
    SLfloat  m_lastTouchTimeSec;  //!< Frame time of the last touch event
    SLfloat  m_lastTouchDownSec;  //!< Time of last touch down
    SLint    m_touchDowns;        //!< No. of finger touchdowns

    // Video stuff
    AVCaptureSession*   m_avSession;            //!< Audio video session
    NSString*           m_avSessionPreset;      //!< Session name
    bool                m_lastVideoImageIsConsumed;
}
@property (strong, nonatomic) EAGLContext *context;
@property (strong, nonatomic) CMMotionManager *motionManager;
@end
//-----------------------------------------------------------------------------
@implementation ViewController

@synthesize context = _context;

- (void)dealloc
{
   [_context release];
   [super dealloc];
}
//-----------------------------------------------------------------------------
- (void)viewDidLoad
{
    [super viewDidLoad];
   
    self.context = [[[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3] autorelease];
    if (!self.context)
    {   NSLog(@"Failed to create ES3 context");
        self.context = [[[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2] autorelease];
        if (!self.context) NSLog(@"Failed to create ES2 context");
    }
    
    myView = (GLKView *)self.view;
    myView.context = self.context;
    myView.drawableDepthFormat = GLKViewDrawableDepthFormat24;
   
    if([UIDevice currentDevice].multitaskingSupported)
       myView.drawableMultisample = GLKViewDrawableMultisample4X;
   
    self.preferredFramesPerSecond = 60;
    self.view.multipleTouchEnabled = true;
    m_touchDowns = 0;
   
    //[self setupGL];
    [EAGLContext setCurrentContext:self.context];
    
    // Init motion manager
    self.motionManager = [[CMMotionManager alloc] init];
    
    // determine device pixel ratio and dots per inch
    screenScale = [UIScreen mainScreen].scale;
    float dpi;
    if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPad)
        dpi = 132 * screenScale;
    else if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPhone)
         dpi = 163 * screenScale;
    else dpi = 160 * screenScale;
   
    SLVstring cmdLineArgs;
    SLstring exeDir = SLFileSystem::getCurrentWorkingDir();
    SLstring configDir = SLFileSystem::getAppsWritableDir();
    
    //////////////////////////
    slCreateScene(cmdLineArgs,
                  exeDir,
                  exeDir,
                  exeDir,
                  exeDir,
                  configDir);
    //////////////////////////
   
    ///////////////////////////////////////////////////////////////////////
    svIndex = slCreateSceneView(self.view.bounds.size.width * screenScale,
                                self.view.bounds.size.height * screenScale,
                                dpi,
                                C_sceneMeshLoad,
                                (void*)&onPaintRTGL,
                                0,
                                0,
                                0);
    ///////////////////////////////////////////////////////////////////////
    
    [self setMotionInterval:1.0/60.0];
    [self setupVideoCapture];
}
//-----------------------------------------------------------------------------
- (void)viewDidUnload
{
    [super viewDidUnload];
   
    slTerminate();
   
    if ([EAGLContext currentContext] == self.context)
    {   [EAGLContext setCurrentContext:nil];
    }
    self.context = nil;
}
//-----------------------------------------------------------------------------
- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Release any cached data, images, etc. that aren't in use.
}
//-----------------------------------------------------------------------------
- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
{
    // Return YES for supported orientations
    if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPhone)
         return (interfaceOrientation != UIInterfaceOrientationPortraitUpsideDown);
    else return YES;
}
//-----------------------------------------------------------------------------
- (void)update
{
    slResize(svIndex, self.view.bounds.size.width  * screenScale,
                      self.view.bounds.size.height * screenScale);
}
//-----------------------------------------------------------------------------
- (void)glkView:(GLKView *)view drawInRect:(CGRect)rect
{
    
    if (slGetVideoType() != VT_NONE)
    {   if (![m_avSession isRunning])
            [m_avSession startRunning];
    } else
    {   if ([m_avSession isRunning])
            [m_avSession stopRunning];
    }
    
    slUpdateAndPaint(svIndex);
    m_lastVideoImageIsConsumed = true;
}
//-----------------------------------------------------------------------------
// touchesBegan receives the finger thouch down events
- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *) event
{
    NSArray* myTouches = [touches allObjects];
    UITouch* touch1 = [myTouches objectAtIndex:0];
    CGPoint pos1 = [touch1 locationInView:touch1.view];
    pos1.x *= screenScale;
    pos1.y *= screenScale;
    float touchDownNowSec = GetSeconds();
   
    // end touch actions on sequential finger touch downs
    if (m_touchDowns > 0)
    {
        if (m_touchDowns == 1)
            slMouseUp(svIndex, MB_left, pos1.x, pos1.y, K_none);
        if (m_touchDowns == 2)
            slTouch2Up(svIndex, 0, 0, 0, 0);
      
        // Reset touch counter if last touch event is older than a second.
        // This resolves the problem off loosing track in touch counting e.g.
        // when somebody touches with the flat hand.
        if (m_lastTouchTimeSec < (m_lastFrameTimeSec - 2.0f))
            m_touchDowns = 0;
    }
   
    m_touchDowns += [touches count];
    //printf("Begin tD: %d, touches count: %d\n", m_touchDowns, [touches count]);
   
    if (m_touchDowns == 1 && [touches count] == 1)
    {   if (touchDownNowSec - m_lastTouchDownSec < 0.3f)
            slDoubleClick(svIndex, MB_left, pos1.x, pos1.y, K_none);
        else
            slMouseDown(svIndex, MB_left, pos1.x, pos1.y, K_none);
    } else
    if (m_touchDowns == 2)
    {
        if ([touches count] == 2)
        {   UITouch* touch2 = [myTouches objectAtIndex:1];
            CGPoint pos2 = [touch2 locationInView:touch2.view];
            pos2.x *= screenScale;
            pos2.y *= screenScale;
            slTouch2Down(svIndex, pos1.x, pos1.y, pos2.x, pos2.y);
        } else
        if ([touches count] == 1) // delayed 2nd finger touch
            slTouch2Down(svIndex, 0, 0, 0, 0);
    }
   
    m_lastTouchTimeSec = m_lastTouchDownSec = touchDownNowSec;
}
//-----------------------------------------------------------------------------
// touchesMoved receives the finger move events
- (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event
{
    NSArray* myTouches = [touches allObjects];
    UITouch* touch1 = [myTouches objectAtIndex:0];
    CGPoint pos1 = [touch1 locationInView:touch1.view];
    pos1.x *= screenScale;
    pos1.y *= screenScale;
   
    if (m_touchDowns == 1 && [touches count] == 1)
    {   slMouseMove(svIndex, pos1.x, pos1.y);
    } else
    if (m_touchDowns == 2 && [touches count] == 2)
    {   UITouch* touch2 = [myTouches objectAtIndex:1];
        CGPoint pos2 = [touch2 locationInView:touch2.view];
        pos2.x *= screenScale;
        pos2.y *= screenScale;
        slTouch2Move(svIndex, pos1.x, pos1.y, pos2.x, pos2.y);
    }
   
    m_lastTouchTimeSec = m_lastFrameTimeSec;
}
//-----------------------------------------------------------------------------
// touchesEnded receives the finger thouch release events
- (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event
{
    NSArray* myTouches = [touches allObjects];
    UITouch* touch1 = [myTouches objectAtIndex:0];
    CGPoint pos1 = [touch1 locationInView:touch1.view];
    pos1.x *= screenScale;
    pos1.y *= screenScale;
   
    if (m_touchDowns == 1 || [touches count] == 1)
    {   slMouseUp(svIndex, MB_left, pos1.x, pos1.y, K_none);
    } else
    if (m_touchDowns == 2 && [touches count] >= 2)
    {   UITouch* touch2 = [myTouches objectAtIndex:1];
        CGPoint pos2 = [touch2 locationInView:touch2.view];
        pos2.x *= screenScale;
        pos2.y *= screenScale;
        slTouch2Up(svIndex, pos1.x, pos1.y, pos2.x, pos2.y);
    }
    m_touchDowns -= (int)[touches count];
    if (m_touchDowns < 0) m_touchDowns = 0;
   
    //printf("End   tD: %d, touches count: %d\n", m_touchDowns, [touches count]);
   
    m_lastTouchTimeSec = m_lastFrameTimeSec;
}
//-----------------------------------------------------------------------------
// touchesCancle receives the cancle event on an iPhone call
- (void)touchesCancle:(NSSet *)touches withEvent:(UIEvent *)event
{
    NSArray* myTouches = [touches allObjects];
    UITouch* touch1 = [myTouches objectAtIndex:0];
    CGPoint pos1 = [touch1 locationInView:touch1.view];
   
    if (m_touchDowns == 1 || [touches count] == 1)
    {   slMouseUp(svIndex, MB_left, pos1.x, pos1.y, K_none);
    } else
    if (m_touchDowns == 2 && [touches count] >= 2)
    {   UITouch* touch2 = [myTouches objectAtIndex:1];
        CGPoint pos2 = [touch2 locationInView:touch2.view];
        slTouch2Up(svIndex, pos1.x, pos1.y, pos2.x, pos2.y);
    }
    m_touchDowns -= (int)[touches count];
    if (m_touchDowns < 0) m_touchDowns = 0;
   
    //printf("End   tD: %d, touches count: %d\n", m_touchDowns, [touches count]);
   
    m_lastTouchTimeSec = m_lastFrameTimeSec;
}
//-----------------------------------------------------------------------------
// Event handler for a new camera image (taken from the GLCameraRipple example)
- (void)captureOutput:(AVCaptureOutput *)captureOutput
        didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
        fromConnection:(AVCaptureConnection *)connection
{
    if (!m_lastVideoImageIsConsumed) return;
        
    CVReturn err;
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    CVPixelBufferLockBaseAddress(pixelBuffer,0);
    
    int width  = (int) CVPixelBufferGetWidth(pixelBuffer);
    int height = (int) CVPixelBufferGetHeight(pixelBuffer);
    unsigned char* data = (unsigned char*)CVPixelBufferGetBaseAddress(pixelBuffer);
        
    if(!data)
    {   NSLog(@"No pixel buffer data");
        return;
    }
        
    slCopyVideoImage(width, height, PF_bgra, data, false);
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
        
    m_lastVideoImageIsConsumed = false;
}
//-----------------------------------------------------------------------------
-(void)onAccelerationData:(CMAcceleration)acceleration
{
    //SLVec3f acc(acceleration.x,acceleration.y,acceleration.z);
    //acc.print("Acc:");
}
//-----------------------------------------------------------------------------
-(void)onGyroData:(CMRotationRate)rotation
{
    //SLVec3f rot(rotation.x,rotation.y,rotation.z);
    //rot.print("Rot:");
}
//-----------------------------------------------------------------------------
-(void)onMotionData:(CMAttitude*)attitude
{
    if (slUsesRotation(svIndex))
    {
        //SLVec3f att(attitude.roll,attitude.pitch,attitude.yaw);
        //att.print("att:");
        slRotationPYR(svIndex, attitude.roll, attitude.yaw, attitude.pitch);
    }
}
//-----------------------------------------------------------------------------
//! Prepares the video capture (taken from the GLCameraRipple example)
- (void)setupVideoCapture
{
    if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPad)
         m_avSessionPreset = AVCaptureSessionPreset640x480;
    else m_avSessionPreset = AVCaptureSessionPreset1280x720;
    
    //-- Setup Capture Session.
    m_avSession = [[AVCaptureSession alloc] init];
    [m_avSession beginConfiguration];
    
    //-- Set preset session size.
    [m_avSession setSessionPreset:m_avSessionPreset];
    
    //-- Creata a video device and input from that Device.  Add the input to the capture session.
    AVCaptureDevice * videoDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    if(videoDevice == nil)
        assert(0);
    
    //-- Add the device to the session.
    NSError *error;
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:videoDevice error:&error];
    if(error)
        assert(0);
    
    [m_avSession addInput:input];
    
    //-- Create the output for the capture session.
    AVCaptureVideoDataOutput * dataOutput = [[AVCaptureVideoDataOutput alloc] init];
    [dataOutput setAlwaysDiscardsLateVideoFrames:YES]; // Probably want to set this to NO when recording
    
    /*
    NSDictionary *formats = [NSDictionary dictionaryWithObjectsAndKeys:
       @"kCVPixelFormatType_1Monochrome", [NSNumber numberWithInt:kCVPixelFormatType_1Monochrome],
       @"kCVPixelFormatType_2Indexed", [NSNumber numberWithInt:kCVPixelFormatType_2Indexed],
       @"kCVPixelFormatType_4Indexed", [NSNumber numberWithInt:kCVPixelFormatType_4Indexed],
       @"kCVPixelFormatType_8Indexed", [NSNumber numberWithInt:kCVPixelFormatType_8Indexed],
       @"kCVPixelFormatType_1IndexedGray_WhiteIsZero", [NSNumber numberWithInt:kCVPixelFormatType_1IndexedGray_WhiteIsZero],
       @"kCVPixelFormatType_2IndexedGray_WhiteIsZero", [NSNumber numberWithInt:kCVPixelFormatType_2IndexedGray_WhiteIsZero],
       @"kCVPixelFormatType_4IndexedGray_WhiteIsZero", [NSNumber numberWithInt:kCVPixelFormatType_4IndexedGray_WhiteIsZero],
       @"kCVPixelFormatType_8IndexedGray_WhiteIsZero", [NSNumber numberWithInt:kCVPixelFormatType_8IndexedGray_WhiteIsZero],
       @"kCVPixelFormatType_16BE555", [NSNumber numberWithInt:kCVPixelFormatType_16BE555],
       @"kCVPixelFormatType_16LE555", [NSNumber numberWithInt:kCVPixelFormatType_16LE555],
       @"kCVPixelFormatType_16LE5551", [NSNumber numberWithInt:kCVPixelFormatType_16LE5551],
       @"kCVPixelFormatType_16BE565", [NSNumber numberWithInt:kCVPixelFormatType_16BE565],
       @"kCVPixelFormatType_16LE565", [NSNumber numberWithInt:kCVPixelFormatType_16LE565],
       @"kCVPixelFormatType_24RGB", [NSNumber numberWithInt:kCVPixelFormatType_24RGB],
       @"kCVPixelFormatType_24BGR", [NSNumber numberWithInt:kCVPixelFormatType_24BGR],
       @"kCVPixelFormatType_32ARGB", [NSNumber numberWithInt:kCVPixelFormatType_32ARGB],
       @"kCVPixelFormatType_32BGRA", [NSNumber numberWithInt:kCVPixelFormatType_32BGRA],
       @"kCVPixelFormatType_32ABGR", [NSNumber numberWithInt:kCVPixelFormatType_32ABGR],
       @"kCVPixelFormatType_32RGBA", [NSNumber numberWithInt:kCVPixelFormatType_32RGBA],
       @"kCVPixelFormatType_64ARGB", [NSNumber numberWithInt:kCVPixelFormatType_64ARGB],
       @"kCVPixelFormatType_48RGB", [NSNumber numberWithInt:kCVPixelFormatType_48RGB],
       @"kCVPixelFormatType_32AlphaGray", [NSNumber numberWithInt:kCVPixelFormatType_32AlphaGray],
       @"kCVPixelFormatType_16Gray", [NSNumber numberWithInt:kCVPixelFormatType_16Gray],
       @"kCVPixelFormatType_422YpCbCr8", [NSNumber numberWithInt:kCVPixelFormatType_422YpCbCr8],
       @"kCVPixelFormatType_4444YpCbCrA8", [NSNumber numberWithInt:kCVPixelFormatType_4444YpCbCrA8],
       @"kCVPixelFormatType_4444YpCbCrA8R", [NSNumber numberWithInt:kCVPixelFormatType_4444YpCbCrA8R],
       @"kCVPixelFormatType_444YpCbCr8", [NSNumber numberWithInt:kCVPixelFormatType_444YpCbCr8],
       @"kCVPixelFormatType_422YpCbCr16", [NSNumber numberWithInt:kCVPixelFormatType_422YpCbCr16],
       @"kCVPixelFormatType_422YpCbCr10", [NSNumber numberWithInt:kCVPixelFormatType_422YpCbCr10],
       @"kCVPixelFormatType_444YpCbCr10", [NSNumber numberWithInt:kCVPixelFormatType_444YpCbCr10],
       @"kCVPixelFormatType_420YpCbCr8Planar", [NSNumber numberWithInt:kCVPixelFormatType_420YpCbCr8Planar],
       @"kCVPixelFormatType_420YpCbCr8PlanarFullRange", [NSNumber numberWithInt:kCVPixelFormatType_420YpCbCr8PlanarFullRange],
       @"kCVPixelFormatType_422YpCbCr_4A_8BiPlanar", [NSNumber numberWithInt:kCVPixelFormatType_422YpCbCr_4A_8BiPlanar],
       @"kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange", [NSNumber numberWithInt:kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange],
       @"kCVPixelFormatType_420YpCbCr8BiPlanarFullRange", [NSNumber numberWithInt:kCVPixelFormatType_420YpCbCr8BiPlanarFullRange],
       @"kCVPixelFormatType_422YpCbCr8_yuvs", [NSNumber numberWithInt:kCVPixelFormatType_422YpCbCr8_yuvs],
       @"kCVPixelFormatType_422YpCbCr8FullRange", [NSNumber numberWithInt:kCVPixelFormatType_422YpCbCr8FullRange],
    nil];

    for (NSNumber *fmt in [dataOutput availableVideoCVPixelFormatTypes])
    {   NSLog(@"%@", [formats objectForKey:fmt]);
    }
    */

    //-- Set to BGRA.
    // Corevideo only supports:
    // kCVPixelFormatType_32BGRA
    // kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
    // kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    [dataOutput setVideoSettings:[NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCVPixelFormatType_32BGRA]
        forKey:(id)kCVPixelBufferPixelFormatTypeKey]];
    
    // Set dispatch to be on the main thread so OpenGL can do things with the data
    [dataOutput setSampleBufferDelegate:self queue:dispatch_get_main_queue()];
    
    [m_avSession addOutput:dataOutput];
    [m_avSession commitConfiguration];
    
    m_lastVideoImageIsConsumed = true;
}
//-----------------------------------------------------------------------------
//! Starts the acceleronometer update if the interval time > 0 else it stops
- (void) setAccelerometerInterval:(double)intervalTimeSEC
{
    if ([self.motionManager isAccelerometerAvailable] == YES)
    {
        if ([self.motionManager isAccelerometerActive] == NO && intervalTimeSEC > 0)
        {    self.motionManager.accelerometerUpdateInterval = intervalTimeSEC;
            [self.motionManager startAccelerometerUpdatesToQueue:[NSOperationQueue currentQueue]
                                    withHandler:^(CMAccelerometerData *accelerometerData, NSError *error)
                                    {   [self onAccelerationData:accelerometerData.acceleration];
                                        if(error){NSLog(@"%@", error);}
                                    }
            ];
        } else [self.motionManager stopAccelerometerUpdates];
    }
}
//-----------------------------------------------------------------------------
//! Starts the gyroscope update if the interval time > 0 else it stops
- (void) setGyroInterval:(double)intervalTimeSEC
{
    if ([self.motionManager isGyroAvailable] == YES)
    {   
        if ([self.motionManager isGyroActive] == NO && intervalTimeSEC > 0)
        {    self.motionManager.gyroUpdateInterval = intervalTimeSEC;
            [self.motionManager startGyroUpdatesToQueue:[NSOperationQueue currentQueue]
                                    withHandler:^(CMGyroData *gyroData, NSError *error)
                                    {   [self onGyroData:gyroData.rotationRate];
                                        if(error){NSLog(@"%@", error);}
                                    }
            ];
        } else [self.motionManager stopGyroUpdates];
    }
}
//-----------------------------------------------------------------------------
//! Starts the motion data update if the interval time > 0 else it stops
- (void) setMotionInterval:(double)intervalTimeSEC
{
    if ([self.motionManager isDeviceMotionAvailable] == YES)
    {
        if ([self.motionManager isAccelerometerActive] == NO && intervalTimeSEC > 0)
        {    self.motionManager.deviceMotionUpdateInterval = intervalTimeSEC;
            [self.motionManager startDeviceMotionUpdatesToQueue:[NSOperationQueue currentQueue]
                                    withHandler:^(CMDeviceMotion *deviceMotion, NSError *error)
                                    {   [self onMotionData:deviceMotion.attitude];
                                        if(error){NSLog(@"%@", error);}
                                    }
            ];
        } else [self.motionManager stopDeviceMotionUpdates];
    }
}
//-----------------------------------------------------------------------------
@end
