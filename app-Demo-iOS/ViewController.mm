//#############################################################################
//  File:      ViewController.m
//  Purpose:   Top level iOS view controller code that interfaces SLProject
//             The demo application demonstrates most features of the SLProject
//             framework. Implementation of the GUI with the GLFW3 framework
//             that can create a window and receive system event on desktop OS
//             such as Windows, MacOS and Linux.
//  Author:    Marcus Hudritsch
//  Date:      November 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

// Objective C imports
#import "ViewController.h"
#import <CoreMotion/CoreMotion.h>

// C++ includes for the SceneLibrary
#include <SLMath.h>
#include <SLFileSystem.h>
#include <SLInterface.h>
#include <SLCVCapture.h>
#include <AppDemoGui.h>
#include <mach/mach_time.h>

// Declaration of scene load function
extern void appDemoLoadScene(SLScene* s, SLSceneView* sv, SLSceneID sceneID);

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
{  [myView display];
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
@interface ViewController () <CLLocationManagerDelegate>
{
    SLfloat  m_lastFrameTimeSec;  //!< Timestamp for passing highres time
    SLfloat  m_lastTouchTimeSec;  //!< Frame time of the last touch event
    SLfloat  m_lastTouchDownSec;  //!< Time of last touch down
    SLint    m_touchDowns;        //!< No. of finger touchdowns

    // Video stuff
    AVCaptureSession*   m_avSession;            //!< Audio video session
    NSString*           m_avSessionPreset;      //!< Session name
    bool                m_lastVideoImageIsConsumed;
    int                 m_lastVideoType;        //! VT_NONE=0,VT_MAIN=1,VT_SCND=2
    bool                m_locationIsRunning;    //! GPS is running
}
@property (strong, nonatomic) EAGLContext       *context;
@property (strong, nonatomic) CMMotionManager   *motionManager;
@property (strong, nonatomic) NSTimer           *motionTimer;
@property (strong, nonatomic) CLLocationManager *locationManager;
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
    
    /////////////////////////////////////////////
    slCreateAppAndScene(cmdLineArgs,
                        exeDir,
                        exeDir,
                        exeDir,
                        exeDir,
                        exeDir,
                        exeDir,
                        configDir,
                        "AppDemo_iOS",
                        (void*)appDemoLoadScene);
    /////////////////////////////////////////////
    
    // This load the GUI configs that are locally stored
    AppDemoGui::loadConfig(dpi);
   
    ///////////////////////////////////////////////////////////////////////
    svIndex = slCreateSceneView(self.view.bounds.size.height * screenScale,
                                self.view.bounds.size.width * screenScale,
                                dpi,
                                SID_Revolver,
                                (void*)&onPaintRTGL,
                                0,
                                0,
                                (void*)AppDemoGui::build);
    ///////////////////////////////////////////////////////////////////////
    
    [self setupMotionManager: 1.0/20.0];
    [self setupLocationManager];
}
//-----------------------------------------------------------------------------
- (void)viewDidUnload
{
    printf("viewDidUnload\n");
    
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
- (void)update
{
    slResize(svIndex, self.view.bounds.size.width  * screenScale,
                      self.view.bounds.size.height * screenScale);
}
//-----------------------------------------------------------------------------
- (void)glkView:(GLKView *)view drawInRect:(CGRect)rect
{
    [self setVideoType:slGetVideoType()];
    
    if (slUsesLocation())
         [self startLocationManager];
    else [self stopLocationManager];
    
    slUpdateAndPaint(svIndex);
    m_lastVideoImageIsConsumed = true;
    
    if (slShouldClose())
    {
        AppDemoGui::saveConfig();
        slTerminate();
        exit(0);
    }
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
    //printf("Begin tD: %d, touches count: %u\n", m_touchDowns, (SLuint)[touches count]);
   
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

    m_touchDowns = 0;
   
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
    // Don't copy the available image if the last wasn't consumed
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
    if (slUsesRotation())
    {
        if ([[UIDevice currentDevice] orientation] == UIDeviceOrientationLandscapeLeft)
        {
            NSLog(@"UIDeviceOrientationLandscapeLeft");
        }
        else if ([[UIDevice currentDevice] orientation ]== UIDeviceOrientationLandscapeRight)
        {
            float pitch = attitude.roll            - SL_HALFPI;
            float yaw   = attitude.yaw             - SL_HALFPI;
            float roll  = attitude.pitch;
            SL_LOG("Pitch: %3.0f, Yaw: %3.0f, Roll: %3.0f\n",
                   pitch*SL_RAD2DEG,
                   yaw*SL_RAD2DEG,
                   roll*SL_RAD2DEG);
        }
        else if([[UIDevice currentDevice] orientation] == UIDeviceOrientationPortrait)
        {
            NSLog(@"UIDeviceOrientationPortrait");
        }
        else if([[UIDevice currentDevice] orientation] == UIDeviceOrientationPortraitUpsideDown )
        {
            NSLog(@"UIDeviceOrientationPortraitUpsideDown");
        }
    }
}
//-----------------------------------------------------------------------------
//! Prepares the video capture (taken from the GLCameraRipple example)
- (void)setupVideo: (bool)useFaceCamera
{
    m_avSessionPreset = AVCaptureSessionPreset640x480;
    
    //-- Setup Capture Session.
    m_avSession = [[AVCaptureSession alloc] init];
    [m_avSession beginConfiguration];
    
    //-- Set preset session size.
    [m_avSession setSessionPreset:m_avSessionPreset];
    
    //-- Creata a video device and input from that Device.  Add the input to the capture session.
    //AVCaptureDevice* videoDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    AVCaptureDevice* videoDevice = nil;
    if (useFaceCamera)
        videoDevice = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera
                                       mediaType:AVMediaTypeVideo
                                       position:AVCaptureDevicePositionFront];
    else
        videoDevice = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera
                                       mediaType:AVMediaTypeVideo
                                       position:AVCaptureDevicePositionBack];
    if(videoDevice == nil)
        assert(0);
    
    /*
    for (AVCaptureDeviceFormat *format in [videoDevice formats] ) {
        CMFormatDescriptionRef description = format.formatDescription;
        CMVideoDimensions dimensions = CMVideoFormatDescriptionGetDimensions(description);
        SL_LOG("%s: %d x %d\n", format.description.UTF8String, dimensions.width, dimensions.height);
    }
    */
    
    //-- Add the device to the session.
    NSError *error;
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:videoDevice error:&error];
    if(error)
        assert(0);
    
    [m_avSession addInput:input];
    
    //-- Create the output for the capture session.
    AVCaptureVideoDataOutput * dataOutput = [[AVCaptureVideoDataOutput alloc] init];
    [dataOutput setAlwaysDiscardsLateVideoFrames:YES]; // Probably want to set this to NO when recording

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
//! Sets the video according to the passed type (0=NONE, 1=Main, 2=Secondary)
/* The main camera on iOS is the back camera and the the secondary is the front
 camera that faces the face.
*/
- (void) setVideoType:(int)videoType
{
    if (videoType == VT_NONE) // No video needed. Turn off any video
    {
        if (m_avSession != nil && ![m_avSession isRunning])
        {   printf("Stopping AV Session\n");
            [m_avSession stopRunning];
        }
    }
    if (videoType == VT_FILE) // Turn off any live video
    {
        if (m_avSession != nil && ![m_avSession isRunning])
        {   printf("Stopping AV Session\n");
            [m_avSession stopRunning];
        }
        SLCVCapture::grabAndAdjustForSL();
    }
    else if (videoType == VT_MAIN) // back facing video needed
    {
        if (m_avSession == nil)
        {   printf("Creating AV Session for Front Camera\n");
            [self setupVideo:false];
            printf("Starting AV Session\n");
            [m_avSession startRunning];
        }
        else if (m_lastVideoType == videoType)
        {
            if (![m_avSession isRunning])
            {   printf("Starting AV Session\n");
                [m_avSession startRunning];
            }
        }
        else
        {   if ([m_avSession isRunning])
            {   printf("Deleting AV Session\n");
                [m_avSession stopRunning];
                m_avSession = nil;
            }
            printf("Creating AV Session for Front Camera\n");
            [self setupVideo:false];
            printf("Starting AV Session\n");
            [m_avSession startRunning];
        }
    }
    else if (videoType == VT_SCND) // Video from selfie camera needed
    {
        if (m_avSession == nil)
        {   printf("Creating AV Session for Back Camera\n");
            [self setupVideo:true];
            printf("Starting AV Session\n");
            [m_avSession startRunning];
        }
        else if (m_lastVideoType == videoType)
        {   if (![m_avSession isRunning])
            {   printf("Starting AV Session\n");
                [m_avSession startRunning];
            }
        }
        else
        {   if ([m_avSession isRunning])
            {   printf("Deleting AV Session\n");
                [m_avSession stopRunning];
                m_avSession = nil;
            }
            printf("Creating AV Session for Back Camera\n");
            [self setupVideo:true];
            printf("Starting AV Session\n");
            [m_avSession startRunning];
        }
    }
    
    m_lastVideoType = videoType;
}
//-----------------------------------------------------------------------------
//! Starts the motion data update if the interval time > 0 else it stops
- (void) setupMotionManager:(double)intervalTimeSEC
{
    // Init motion manager
    self.motionManager = [[CMMotionManager alloc] init];
    
    if ([self.motionManager isDeviceMotionAvailable] == YES)
    {
        self.motionManager.deviceMotionUpdateInterval = intervalTimeSEC;
        
        // See also: https://developer.apple.com/documentation/coremotion/getting_processed_device_motion_data/understanding_reference_frames_and_device_attitude?language=objc
        [self.motionManager startDeviceMotionUpdatesUsingReferenceFrame:CMAttitudeReferenceFrameXMagneticNorthZVertical
                                                                toQueue:[NSOperationQueue currentQueue]
                                                            withHandler: ^(CMDeviceMotion *motion, NSError *error){
                                                                [self performSelectorOnMainThread:@selector(onDeviceMotionUpdate:)
                                                                                       withObject:motion waitUntilDone:YES];
                                                            }];
    } else [self.motionManager stopDeviceMotionUpdates];
}
//-----------------------------------------------------------------------------
- (void)onDeviceMotionUpdate:(CMDeviceMotion*)motion
{
    if (slUsesRotation())
    {
        CMDeviceMotion *motionData = self.motionManager.deviceMotion;
        CMAttitude *attitude = motionData.attitude;
        
        //Get sensor rotation as quaternion. This quaternion describes a rotation
        //relative to NWU-frame
        //(see: https://developer.apple.com/documentation/coremotion/getting_processed_device_motion_data/understanding_reference_frames_and_device_attitude)
        CMQuaternion q = attitude.quaternion;
        
        //Add rotation of 90 deg. around z-axis to relate the sensor rotation to an ENU-frame (as in Android)
        GLKQuaternion qNWU    = GLKQuaternionMake(q.x, q.y, q.z, q.w);
        GLKQuaternion qRot90Z = GLKQuaternionMakeWithAngleAndAxis(GLKMathDegreesToRadians(90), 0, 0, 1);
        GLKQuaternion qENU    = GLKQuaternionMultiply(qRot90Z, qNWU);
        
        // Send quaternion to SLProject
        slRotationQUAT(qENU.q[0], qENU.q[1], qENU.q[2], qENU.q[3]);
        
        // See the following routines how the rotation is used:
        // SLScene::onRotationQUAT calculates the offset if _zeroYawAtStart is true
        // SLCamera::setView how the device rotation is processed for the camera's view
    }
}
//-----------------------------------------------------------------------------
//! Starts the location data update if the interval time > 0 else it stops
- (void) setupLocationManager
{
    if ([CLLocationManager locationServicesEnabled])
    {
        // Init location manager
        self.locationManager = [[CLLocationManager alloc] init];
        self.locationManager.delegate = self;
        self.locationManager.desiredAccuracy = kCLLocationAccuracyBest;
        //self.locationManager.distanceFilter = 1;
        
        // for iOS 8, specific user level permission is required,
        // "when-in-use" authorization grants access to the user's location.
        // important: be sure to include NSLocationWhenInUseUsageDescription along with its
        // explanation string in your Info.plist or startUpdatingLocation will not work
        if ([self.locationManager respondsToSelector:@selector(requestWhenInUseAuthorization)]) {
            [self.locationManager requestWhenInUseAuthorization];
        }
    } else {
        /* Location services are not enabled.
         Take appropriate action: for instance, prompt the
         user to enable the location services */
        NSLog(@"Location services are not enabled");
    }
    
    m_locationIsRunning = false;
        
}
//-----------------------------------------------------------------------------
//! Starts the location data update
- (void) startLocationManager
{
    if (!m_locationIsRunning)
    {
        [self.locationManager startUpdatingLocation];
        m_locationIsRunning = true;
        printf("Starting Location Manager\n");
    }
}
//-----------------------------------------------------------------------------
//! Stops the location data update
- (void) stopLocationManager
{
    if (m_locationIsRunning)
    {
        [self.locationManager stopUpdatingLocation];
        m_locationIsRunning = false;
        printf("Stopping Location Manager\n");
    }
}
//-----------------------------------------------------------------------------
-(void)locationManager:(CLLocationManager *)manager didUpdateToLocation:(CLLocation *)newLocation fromLocation:(CLLocation *)oldLocation
{
    printf("horizontalAccuracy: %f\n", newLocation.horizontalAccuracy);
    
    // negative horizontal accuracy means no location fix
    if (newLocation.horizontalAccuracy > 0.0)
    {
        slLocationLLA(newLocation.coordinate.latitude,
                      newLocation.coordinate.longitude,
                      newLocation.altitude,
                      newLocation.horizontalAccuracy);
    }
}
//-----------------------------------------------------------------------------
- (void)locationManager:(CLLocationManager *)manager didFailWithError:(NSError *)error {
    // The location "unknown" error simply means the manager is currently unable to get the location.
    // We can ignore this error for the scenario of getting a single location fix, because we already have a
    // timeout that will stop the location manager to save power.
    //
    if ([error code] != kCLErrorLocationUnknown) {
        printf("**** locationManager didFailWithError ****\n");
        [self stopLocationManager];
    }
}
//-----------------------------------------------------------------------------
@end
