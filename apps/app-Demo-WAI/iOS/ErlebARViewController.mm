//#############################################################################
//  File:      ErlebARViewController.m
//  Purpose:
//  Author:    Michael Göttlicher
//  Date:      June 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

//imports
#import "ErlebARViewController.h"
//#import <CoreMotion/CoreMotion.h>
#import <sys/utsname.h>
#import <mach-o/arch.h>
#import <iOS/SENSiOSCamera.h>
//includes
#include <mach/mach_time.h>
#include <Utils.h>
#include "Utils_iOS.h"
#include <ErlebARApp.h>

//-----------------------------------------------------------------------------
@interface ErlebARViewController ()
{
@private
    float  _lastFrameTimeSec;  // Timestamp for passing highres time
    float  _lastTouchTimeSec;  // Frame time of the last touch event
    float  _lastTouchDownSec;  // Time of last touch down
    int    _touchDowns;        // No. of finger touchdowns

    // Global screen scale (2.0 for retina, 1.0 else)
    float _screenScale;
    // ErlebAR app instance
    ErlebARApp _erlebARApp;
    //std::unique_ptr<SENSiOSCamera> _camera;
    SENSiOSCamera* _camera;
}
- (float)getSeconds;

@property (strong, nonatomic) EAGLContext       *context;

@end

//-----------------------------------------------------------------------------
@implementation ErlebARViewController

//-----------------------------------------------------------------------------
- (id)init:(NSString *)nibNameOrNil
{
    self = [self initWithNibName:nibNameOrNil bundle:nil];
    
    if(self)
    {
        _lastFrameTimeSec = 0.f; //todo: this variable remains 0, its never assigned a new value...
        _lastTouchTimeSec = 0.f;
        _lastTouchDownSec = 0.f;
        _touchDowns = 0;
        
        _screenScale = 1.0f;
        _camera = nullptr;
    }
    
    return self;
}
//-----------------------------------------------------------------------------
//(called only on app fresh startup after termination)
- (void)viewDidLoad
{
    printf("viewDidLoad\n");
    [super viewDidLoad];
   
    self.context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3];
    if (!self.context)
    {
        NSLog(@"Failed to create ES3 context");
        self.context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
        if (!self.context)
            NSLog(@"Failed to create ES2 context");
    }
    
    GLKView* view = (GLKView *)self.view;
    view.context = self.context;
    view.drawableDepthFormat = GLKViewDrawableDepthFormat24;
   
    if([UIDevice currentDevice].multitaskingSupported)
       view.drawableMultisample = GLKViewDrawableMultisample4X;
   
    self.preferredFramesPerSecond = 60;
    self.view.multipleTouchEnabled = true;
    _touchDowns = 0;
   
    //[self setupGL];
    [EAGLContext setCurrentContext:self.context];
    
    // determine device pixel ratio and dots per inch
    _screenScale = [UIScreen mainScreen].scale;
    float dpi;
    if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPad)
         dpi = 132 * _screenScale;
    else if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPhone)
         dpi = 163 * _screenScale;
    else dpi = 160 * _screenScale;
      
    // Some computer informations
    struct utsname systemInfo; uname(&systemInfo);
    NSString* model = [NSString stringWithCString:systemInfo.machine encoding:NSUTF8StringEncoding];
    NSString* osver = [[UIDevice currentDevice] systemVersion];
    const NXArchInfo* archInfo = NXGetLocalArchInfo();
    NSString* arch = [NSString stringWithUTF8String:archInfo->description];
    
    Utils::ComputerInfos::model = std::string([model UTF8String]);
    Utils::ComputerInfos::osVer = std::string([osver UTF8String]);
    Utils::ComputerInfos::arch  = std::string([arch UTF8String]);
    
    std::string exePath = Utils_iOS::getCurrentWorkingDir();
    std::string configPath = Utils_iOS::getAppsWritableDir();
    
    Utils::initFileLog(configPath + "log/", true);
    
    _camera = new SENSiOSCamera();
    _erlebARApp.init(self.view.bounds.size.height * _screenScale,
                       self.view.bounds.size.width * _screenScale,
                       dpi,
                       exePath + "data/",
                       configPath,
                       _camera);
}
//-----------------------------------------------------------------------------
- (void)appWillResignActive
{
    _erlebARApp.hold();
    _erlebARApp.update();
}
//-----------------------------------------------------------------------------
- (void)appDidEnterBackground
{
}
//-----------------------------------------------------------------------------
//(not called on startup but only if app was in background)
- (void)appWillEnterForeground
{
    [EAGLContext setCurrentContext:self.context];
    _erlebARApp.resume();
}
//-----------------------------------------------------------------------------
- (void)appDidBecomeActive
{
}
//-----------------------------------------------------------------------------
- (void)appWillTerminate
{
    _erlebARApp.destroy();
    _erlebARApp.update();
    delete _camera;
}
//-----------------------------------------------------------------------------
- (void)didReceiveMemoryWarning
{
    printf("didReceiveMemoryWarning\n");
    
    [super didReceiveMemoryWarning];
   
    _erlebARApp.destroy();
    _erlebARApp.update();
    delete _camera;
    
    if ([EAGLContext currentContext] == self.context)
    {
        [EAGLContext setCurrentContext:nil];
    }
    self.context = nil;
}
//-----------------------------------------------------------------------------
- (void)update
{
    //printf("update\n");
    /*
    slResize(svIndex, self.view.bounds.size.width  * _screenScale,
                      self.view.bounds.size.height * _screenScale);
     */
}
//-----------------------------------------------------------------------------
- (void)glkView:(GLKView *)view drawInRect:(CGRect)rect
{
    //printf("drawInRect: fps: %f\n", (float)self.framesPerSecond);
    _erlebARApp.update();
}
//-----------------------------------------------------------------------------
// touchesBegan receives the finger thouch down events
- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *) event
{
    NSArray* myTouches = [touches allObjects];
    UITouch* touch1 = [myTouches objectAtIndex:0];
    CGPoint pos1 = [touch1 locationInView:touch1.view];
    pos1.x *= _screenScale;
    pos1.y *= _screenScale;
    float touchDownNowSec = [self getSeconds];
   
    // end touch actions on sequential finger touch downs
    if (_touchDowns > 0)
    {
        if (_touchDowns == 1)
            _erlebARApp.mouseUp(0, MB_left, pos1.x, pos1.y, K_none);
        if (_touchDowns == 2)
            _erlebARApp.touch2Up(0, 0, 0, 0, 0);
      
        // Reset touch counter if last touch event is older than a second.
        // This resolves the problem off loosing track in touch counting e.g.
        // when somebody touches with the flat hand.
        if (_lastTouchTimeSec < (_lastFrameTimeSec - 2.0f))
            _touchDowns = 0;
    }
   
    _touchDowns += [touches count];
    //printf("Begin tD: %d, touches count: %u\n", _touchDowns, (SLuint)[touches count]);
   
    if (_touchDowns == 1 && [touches count] == 1)
    {
        if (touchDownNowSec - _lastTouchDownSec < 0.3f)
            _erlebARApp.doubleClick(0, MB_left, pos1.x, pos1.y, K_none);
        else
            _erlebARApp.mouseDown(0, MB_left, pos1.x, pos1.y, K_none);
    }
    else if (_touchDowns == 2)
    {
        if ([touches count] == 2)
        {
            UITouch* touch2 = [myTouches objectAtIndex:1];
            CGPoint pos2 = [touch2 locationInView:touch2.view];
            pos2.x *= _screenScale;
            pos2.y *= _screenScale;
            _erlebARApp.touch2Down(0, pos1.x, pos1.y, pos2.x, pos2.y);
        }
        else if ([touches count] == 1) // delayed 2nd finger touch
            _erlebARApp.touch2Down(0, 0, 0, 0, 0);
    }
   
    _lastTouchTimeSec = _lastTouchDownSec = touchDownNowSec;
}
//-----------------------------------------------------------------------------
// touchesMoved receives the finger move events
- (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event
{
    NSArray* myTouches = [touches allObjects];
    UITouch* touch1 = [myTouches objectAtIndex:0];
    CGPoint pos1 = [touch1 locationInView:touch1.view];
    pos1.x *= _screenScale;
    pos1.y *= _screenScale;
   
    if (_touchDowns == 1 && [touches count] == 1)
    {
        _erlebARApp.mouseMove(0, pos1.x, pos1.y);
    }
    else if (_touchDowns == 2 && [touches count] == 2)
    {
        UITouch* touch2 = [myTouches objectAtIndex:1];
        CGPoint pos2 = [touch2 locationInView:touch2.view];
        pos2.x *= _screenScale;
        pos2.y *= _screenScale;
        _erlebARApp.touch2Move(0, pos1.x, pos1.y, pos2.x, pos2.y);
    }
   
    _lastTouchTimeSec = _lastFrameTimeSec;
}
//-----------------------------------------------------------------------------
// touchesEnded receives the finger thouch release events
- (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event
{
    NSArray* myTouches = [touches allObjects];
    UITouch* touch1 = [myTouches objectAtIndex:0];
    CGPoint pos1 = [touch1 locationInView:touch1.view];
    pos1.x *= _screenScale;
    pos1.y *= _screenScale;
   
    if (_touchDowns == 1 || [touches count] == 1)
    {
        _erlebARApp.mouseUp(0, MB_left, pos1.x, pos1.y, K_none);
    }
    else if (_touchDowns == 2 && [touches count] >= 2)
    {
        UITouch* touch2 = [myTouches objectAtIndex:1];
        CGPoint pos2 = [touch2 locationInView:touch2.view];
        pos2.x *= _screenScale;
        pos2.y *= _screenScale;
        _erlebARApp.touch2Up(0, pos1.x, pos1.y, pos2.x, pos2.y);
    }

    _touchDowns = 0;
   
    //printf("End   tD: %d, touches count: %d\n", _touchDowns, [touches count]);
   
    _lastTouchTimeSec = _lastFrameTimeSec;
}
//-----------------------------------------------------------------------------
// touchesCancle receives the cancle event on an iPhone call
- (void)touchesCancle:(NSSet *)touches withEvent:(UIEvent *)event
{
    NSArray* myTouches = [touches allObjects];
    UITouch* touch1 = [myTouches objectAtIndex:0];
    CGPoint pos1 = [touch1 locationInView:touch1.view];
   
    if (_touchDowns == 1 || [touches count] == 1)
    {
        _erlebARApp.mouseUp(0, MB_left, pos1.x, pos1.y, K_none);
    }
    else if (_touchDowns == 2 && [touches count] >= 2)
    {
        UITouch* touch2 = [myTouches objectAtIndex:1];
        CGPoint pos2 = [touch2 locationInView:touch2.view];
        _erlebARApp.touch2Up(0, pos1.x, pos1.y, pos2.x, pos2.y);
    }
    _touchDowns -= (int)[touches count];
    if (_touchDowns < 0)
        _touchDowns = 0;
   
    //printf("End   tD: %d, touches count: %d\n", _touchDowns, [touches count]);
   
    _lastTouchTimeSec = _lastFrameTimeSec;
}
//-----------------------------------------------------------------------------
- (float)getSeconds
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
@end
