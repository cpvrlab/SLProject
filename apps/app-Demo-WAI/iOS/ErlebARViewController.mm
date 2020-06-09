//#############################################################################
//  File:      ErlebARViewController.m
//  Purpose:   Top level iOS view controller code that interfaces SLProject
//             The demo application demonstrates most features of the SLProject
//             framework. Implementation of the GUI with the GLFW3 framework
//             that can create a window and receive system event on desktop OS
//             such as Windows, MacOS and Linux.
//  Author:    Marcus Hudritsch
//  Date:      November 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

// Objective C imports
#import "ErlebARViewController.h"
#import <CoreMotion/CoreMotion.h>

// C++ includes for the SceneLibrary
#include <Utils.h>
#include "Utils_iOS.h"

#include <mach/mach_time.h>
#import <sys/utsname.h>
#import <mach-o/arch.h>

#include <ErlebARApp.h>

// Forward declaration of C functions in other files
//extern void appDemoLoadScene(SLProjectScene* s, SLSceneView* sv, SLSceneID sceneID);
//extern bool onUpdateVideo();

//-----------------------------------------------------------------------------
// C-Prototypes
float GetSeconds();
//SLbool onPaintRTGL();
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
@interface ErlebARViewController ()
{
    float  m_lastFrameTimeSec;  //!< Timestamp for passing highres time
    float  m_lastTouchTimeSec;  //!< Frame time of the last touch event
    float  m_lastTouchDownSec;  //!< Time of last touch down
    int    m_touchDowns;        //!< No. of finger touchdowns

    // Video stuff
    AVCaptureSession*   m_avSession;            //!< Audio video session
    NSString*           m_avSessionPreset;      //!< Session name
    bool                m_lastVideoImageIsConsumed;
    int                 m_lastVideoType;        //! VT_NONE=0,VT_MAIN=1,VT_SCND=2
    int                 m_lastVideoSizeIndex;   //! 0=1920x1080, 1=1280x720 else 640x480
    bool                m_locationIsRunning;    //! GPS is running
    
    ErlebARApp* m_erlebARApp;
}
@property (strong, nonatomic) EAGLContext       *context;
//@property (strong, nonatomic) CMMotionManager   *motionManager;
@property (strong, nonatomic) NSTimer           *motionTimer;
//@property (strong, nonatomic) CLLocationManager *locationManager;
@end

//-----------------------------------------------------------------------------
@implementation ErlebARViewController
//I think this is not necessary, it is done automatically if we dont synthesize getters and setters
//@synthesize context = _context;

- (void)dealloc
{
}
//-----------------------------------------------------------------------------
- (void)viewDidLoad
{
    [super viewDidLoad];
   
    self.context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3];
    if (!self.context)
    {   NSLog(@"Failed to create ES3 context");
        self.context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
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
   
    //SLVstring cmdLineArgs;
    std::string exePath = Utils_iOS::getCurrentWorkingDir();
    std::string configPath = Utils_iOS::getAppsWritableDir();
    
    // Some some computer informations
    struct utsname systemInfo; uname(&systemInfo);
    NSString* model = [NSString stringWithCString:systemInfo.machine encoding:NSUTF8StringEncoding];
    NSString* osver = [[UIDevice currentDevice] systemVersion];
    const NXArchInfo* archInfo = NXGetLocalArchInfo();
    NSString* arch = [NSString stringWithUTF8String:archInfo->description];
    
    Utils::ComputerInfos::model = std::string([model UTF8String]);
    Utils::ComputerInfos::osVer = std::string([osver UTF8String]);
    Utils::ComputerInfos::arch  = std::string([arch UTF8String]);
    
    m_erlebARApp = new ErlebARApp();
    m_erlebARApp->init(self.view.bounds.size.height * screenScale,
                       self.view.bounds.size.width * screenScale,
                       dpi,
                       exePath + "data/",
                       configPath,
                       nullptr);
    //SLApplication::calibIniPath  = SLApplication::exePath + "data/calibrations/"; // for calibInitPath
    //Utils::dumpFileSystemRec("SLProject", SLApplication::exePath);
    
    /////////////////////////////////////////////
    /*
    slCreateAppAndScene(cmdLineArgs,
                        SLApplication::exePath,
                        SLApplication::exePath,
                        SLApplication::exePath,
                        SLApplication::exePath,
                        SLApplication::configPath,
                        "AppDemo_iOS",
                        (void*)appDemoLoadScene);
   
    ///////////////////////////////////////////////////////////////////////
    svIndex = slCreateSceneView(SLApplication::scene,self.view.bounds.size.height * screenScale,
                                self.view.bounds.size.width * screenScale,
                                dpi,
                                SID_Revolver,
                                (void*)&onPaintRTGL,
                                0,
                                (void*)createAppDemoSceneView,
                                (void*)AppDemoGui::build,
								(void*)AppDemoGui::loadConfig,
                                (void*)AppDemoGui::saveConfig);
     */
    ///////////////////////////////////////////////////////////////////////
    
    //[self setupMotionManager: 1.0/20.0];
    //[self setupLocationManager];
    
    // Set the available capture resolutions
    
    //CVCapture::instance()->setCameraSize(0, 3, 1920, 1080);
    //CVCapture::instance()->setCameraSize(1, 3, 1280,  720);
    //CVCapture::instance()->setCameraSize(2, 3,  640,  480);
    m_lastVideoSizeIndex = -1; // default size index
}
//-----------------------------------------------------------------------------
- (void)didReceiveMemoryWarning
{
    printf("didReceiveMemoryWarning\n");
    
    [super didReceiveMemoryWarning];
   
    //slTerminate();
   
    if ([EAGLContext currentContext] == self.context)
    {
        [EAGLContext setCurrentContext:nil];
    }
    self.context = nil;
    //[super dealloc];
}
//-----------------------------------------------------------------------------
- (void)update
{
    /*
    slResize(svIndex, self.view.bounds.size.width  * screenScale,
                      self.view.bounds.size.height * screenScale);
     */
}
//-----------------------------------------------------------------------------
- (void)glkView:(GLKView *)view drawInRect:(CGRect)rect
{
    /*
    [self setVideoType:CVCapture::instance()->videoType()
          videoSizeIndex:CVCapture::instance()->activeCamera->camSizeIndex()];
    
    if (slUsesLocation())
         [self startLocationManager];
    else
        [self stopLocationManager];
    */
    
    /////////////////////////////////////////////
    //bool trackingGotUpdated = onUpdateVideo();
    //bool jobIsRunning       = slUpdateParallelJob();
    //bool viewsNeedsRepaint  = slPaintAllViews();
    /////////////////////////////////////////////
    
    m_erlebARApp->update();
    //m_lastVideoImageIsConsumed = true;
    
    /*
    if (slShouldClose())
    {
        slTerminate();
        exit(0);
    }
     */
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
            ;//slMouseUp(svIndex, MB_left, pos1.x, pos1.y, K_none);
        if (m_touchDowns == 2)
            ;//slTouch2Up(svIndex, 0, 0, 0, 0);
      
        // Reset touch counter if last touch event is older than a second.
        // This resolves the problem off loosing track in touch counting e.g.
        // when somebody touches with the flat hand.
        if (m_lastTouchTimeSec < (m_lastFrameTimeSec - 2.0f))
            m_touchDowns = 0;
    }
   
    m_touchDowns += [touches count];
    //printf("Begin tD: %d, touches count: %u\n", m_touchDowns, (SLuint)[touches count]);
   
    if (m_touchDowns == 1 && [touches count] == 1)
    {
        if (touchDownNowSec - m_lastTouchDownSec < 0.3f)
            ;//slDoubleClick(svIndex, MB_left, pos1.x, pos1.y, K_none);
        else
            ;//slMouseDown(svIndex, MB_left, pos1.x, pos1.y, K_none);
    }
    else if (m_touchDowns == 2)
    {
        if ([touches count] == 2)
        {
            UITouch* touch2 = [myTouches objectAtIndex:1];
            CGPoint pos2 = [touch2 locationInView:touch2.view];
            pos2.x *= screenScale;
            pos2.y *= screenScale;
            //slTouch2Down(svIndex, pos1.x, pos1.y, pos2.x, pos2.y);
        }
        else if ([touches count] == 1) // delayed 2nd finger touch
            ;//slTouch2Down(svIndex, 0, 0, 0, 0);
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
    {
        //slMouseMove(svIndex, pos1.x, pos1.y);
    }
    else if (m_touchDowns == 2 && [touches count] == 2)
    {
        UITouch* touch2 = [myTouches objectAtIndex:1];
        CGPoint pos2 = [touch2 locationInView:touch2.view];
        pos2.x *= screenScale;
        pos2.y *= screenScale;
        //slTouch2Move(svIndex, pos1.x, pos1.y, pos2.x, pos2.y);
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
    {
        //slMouseUp(svIndex, MB_left, pos1.x, pos1.y, K_none);
    }
    else if (m_touchDowns == 2 && [touches count] >= 2)
    {
        UITouch* touch2 = [myTouches objectAtIndex:1];
        CGPoint pos2 = [touch2 locationInView:touch2.view];
        pos2.x *= screenScale;
        pos2.y *= screenScale;
        //slTouch2Up(svIndex, pos1.x, pos1.y, pos2.x, pos2.y);
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
    {
        //slMouseUp(svIndex, MB_left, pos1.x, pos1.y, K_none);
    }
    else if (m_touchDowns == 2 && [touches count] >= 2)
    {
        UITouch* touch2 = [myTouches objectAtIndex:1];
        CGPoint pos2 = [touch2 locationInView:touch2.view];
        //slTouch2Up(svIndex, pos1.x, pos1.y, pos2.x, pos2.y);
    }
    m_touchDowns -= (int)[touches count];
    if (m_touchDowns < 0)
        m_touchDowns = 0;
   
    //printf("End   tD: %d, touches count: %d\n", m_touchDowns, [touches count]);
   
    m_lastTouchTimeSec = m_lastFrameTimeSec;
}
//-----------------------------------------------------------------------------
@end
