//#############################################################################
//  File:      SL/SLInterface.cpp
//  Purpose:   Implementation of the main Scene Library C-Interface.
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLInterface.h>
#include <SLApplication.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLAssimpImporter.h>
#include <SLInputManager.h>
#include <SLCVCapture.h>
#include <SLCVCalibration.h>
//#include <SLDemoGui.h>

//! \file SLInterface.cpp SLProject C-functions interface implementation.
/*! \file SLInterface.cpp
The SLInterface.cpp has all implementations of the SLProject C-Interface.
Only these functions should called by the OS-dependent GUI applications.
These functions can be called from any C, C++ or ObjectiveC GUI framework or
by a native API such as Java Native Interface (JNI).
*/

//-----------------------------------------------------------------------------
//! global flag that determines if the application should be closed
bool gShouldClose = false;

//-----------------------------------------------------------------------------
/*! Global creation function for a SLScene instance. This function should be
called only once per application. The SLScene constructor call is delayed until
the first SLSceneView is created to guarantee, that the OpenGL context is
present.<br>
/param cmdLineArgs Command line arguments (not used yet)
/param shaderPath Path to the shader files (readonly)
/param modelPath Path to the 3D model files (readonly)
/param texturePath Path to the texture image files (readonly)
/param fontPath Path to the font image files (readonly)
/param calibrationPath Path to the calibration ini files (readonly)
/param configPath Path where the config files are stored (read-write)
/param applicationName The apps name
/param onSceneLoadCallback C Callback function as void* pointer for the scene creation.
<br>
See examples usages in:
  - app-Demo-GLFW:    AppDemoMainGLFW.cpp in function main()
  - app-Demo-Android: native-lib.cpp      in Java_ch_fhnw_comgr_GLES3Lib_onInit()
  - app-Demo-iOS:     ViewController.m    in viewDidLoad()
*/
void slCreateAppAndScene(SLVstring& cmdLineArgs,
                         SLstring shaderPath,
                         SLstring modelPath,
                         SLstring texturePath,
                         SLstring videoPath,
                         SLstring fontPath,
                         SLstring calibrationPath,
                         SLstring configPath,
                         SLstring applicationName,
                         void*    onSceneLoadCallback)
{
    assert(SLApplication::scene==nullptr && "SLScene is already created!");
   
    // Default paths for all loaded resources
    SLGLProgram::defaultPath      = shaderPath;
    SLGLTexture::defaultPath      = texturePath;
    SLGLTexture::defaultPathFonts = fontPath;
    SLAssimpImporter::defaultPath = modelPath;
    SLCVCapture::videoDefaultPath = videoPath;
    SLCVCalibration::calibIniPath = calibrationPath;
    SLApplication::configPath     = configPath;

    SLGLState* stateGL            = SLGLState::getInstance();
    
    SL_LOG("Path to Models  : %s\n", modelPath.c_str());
    SL_LOG("Path to Shaders : %s\n", shaderPath.c_str());
    SL_LOG("Path to Textures: %s\n", texturePath.c_str());
    SL_LOG("Path to Textures: %s\n", videoPath.c_str());
    SL_LOG("Path to Fonts   : %s\n", fontPath.c_str());
    SL_LOG("Path to Calibr. : %s\n", calibrationPath.c_str());
    SL_LOG("Path to Config. : %s\n", configPath.c_str());
    SL_LOG("OpenCV Version  : %d.%d.%d\n", CV_MAJOR_VERSION,
                                           CV_MINOR_VERSION,
                                           CV_VERSION_REVISION);
    SL_LOG("CV has OpenCL   : %s\n", cv::ocl::haveOpenCL() ? "yes":"no");
    SL_LOG("OpenGL Version  : %s\n", stateGL->glVersion().c_str());
    SL_LOG("Vendor          : %s\n", stateGL->glVendor().c_str());
    SL_LOG("Renderer        : %s\n", stateGL->glRenderer().c_str());
    SL_LOG("GLSL Version    : %s (%s) \n", stateGL->glSLVersion().c_str(),
                                           stateGL->getSLVersionNO().c_str());
    SL_LOG("------------------------------------------------------------------\n");

    SLApplication::createAppAndScene(applicationName, onSceneLoadCallback);
}
//-----------------------------------------------------------------------------
/*! Global creation function for a SLSceneview instance returning the index of 
the sceneview. It creates the new SLSceneView instance by calling the callback
function slNewSceneView. If you have a custom SLSceneView inherited class you 
have to provide a similar function and pass it function pointer to 
slCreateSceneView. You can create multiple sceneview per application.<br>
<br>
See examples usages in:
  - app-Demo-GLFW:    AppDemoMainGLFW.cpp   in function main()
  - app-Demo-Android: AppDemoAndroidJNI.cpp in Java_ch_fhnw_comgr_GLES3Lib_onInit()
  - app-Demo-iOS:     ViewController.m      in viewDidLoad()
*/
int slCreateSceneView(int screenWidth,
                      int screenHeight,
                      int dotsPerInch,
                      SLSceneID initScene,
                      void* onWndUpdateCallback,
                      void* onSelectNodeMeshCallback,
                      void* onNewSceneViewCallback,
                      void* onImGuiBuild)
{
    assert(SLApplication::scene && "No SLApplication::scene!");

    // Use our own sceneview creator callback or the the passed one.
    cbOnNewSceneView newSVCallback;
    if (onNewSceneViewCallback==nullptr)
         newSVCallback = &slNewSceneView;
    else newSVCallback = (cbOnNewSceneView)onNewSceneViewCallback;

    // Create the sceneview & get the pointer with the sceneview index
    SLuint index = newSVCallback();
    SLSceneView* sv = SLApplication::scene->sv(index);

    sv->init("SceneView",
             screenWidth, 
             screenHeight, 
             onWndUpdateCallback,
             onSelectNodeMeshCallback,
             onImGuiBuild);

    // Set default font sizes depending on the dpi no matter if ImGui is used
    if (!SLApplication::dpi) SLApplication::dpi = dotsPerInch;

    // Load GUI fonts depending on the resolution
    sv->gui().loadFonts(SLGLImGui::fontPropDots, SLGLImGui::fontFixedDots);

    // Set active sceneview and load scene. This is done for the first sceneview
    if (!SLApplication::scene->root3D())
    {   if (SLApplication::sceneID == SID_Empty)
             SLApplication::scene->onLoad(SLApplication::scene, sv, initScene);
        else SLApplication::scene->onLoad(SLApplication::scene, sv, SLApplication::sceneID);
    } else sv->onInitialize();
   
    // return the identifier index
    return sv->index();
}
//-----------------------------------------------------------------------------
/*! Global sceneview construction function returning the index of the created
sceneview instance. If you have a custom SLSceneView inherited class you 
have to provide a similar function and pass it function pointer to 
slCreateSceneView.
*/
int slNewSceneView()
{
    SLSceneView* sv = new SLSceneView();
    return sv->index();
}
//-----------------------------------------------------------------------------
/*! Global closing function that deallocates the sceneview and scene instances.
All the scenegraph deallocation is started from here and has to be done before
the GUI app terminates.
*/
bool slShouldClose()
{
    return gShouldClose;
}
//-----------------------------------------------------------------------------
/*! Global closing function that sets our global running flag. This lets
the windowing system know that we want to terminate.
*/
void slShouldClose(bool val)
{
    gShouldClose = val;
}
//-----------------------------------------------------------------------------
/*! Global closing function that deallocates the sceneview and scene instances.
All the scenegraph deallocation is started from here and has to be done before
the GUI app terminates.
*/
void slTerminate()
{
    // Deletes all remaining sceneviews the current scene instance  
    SLApplication::deleteAppAndScene();
}
//-----------------------------------------------------------------------------
/*! Global rendering function that first updates the scene due to user or
device inputs and due to active animations. This happens only if all sceneviews
where finished with rendering. After the update sceneviews onPaint routine is
called to initiate the rendering of the frame. If either the onUpdate or onPaint
returned true a new frame should be drawn.
*/
bool slUpdateAndPaint(int sceneViewIndex)
{  
    SLSceneView* sv = SLApplication::scene->sv(sceneViewIndex);

    bool sceneGotUpdated = SLApplication::scene->onUpdate();
    
    bool viewNeedsUpdate =  sv->onPaint();
    
    return sceneGotUpdated || viewNeedsUpdate;
}
//-----------------------------------------------------------------------------
/*! Global resize function that must be called whenever the OpenGL frame
changes it's size.
*/
void slResize(int sceneViewIndex, int width, int height)
{
    SLResizeEvent* e = new SLResizeEvent;
    e->svIndex = sceneViewIndex;
    e->width = width;
    e->height = height;
    SLApplication::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse button down events. 
*/
void slMouseDown(int sceneViewIndex, SLMouseButton button, 
                 int xpos, int ypos, SLKey modifier) 
{
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseDown);
    e->svIndex = sceneViewIndex;
    e->button = button;
    e->x = xpos;
    e->y = ypos;
    e->modifier = modifier;
    SLApplication::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse move events.
*/
void slMouseMove(int sceneViewIndex, int x, int y)
{  
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseMove);
    e->svIndex = sceneViewIndex;
    e->x = x;
    e->y = y;
    SLApplication::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse button up events.
*/
void slMouseUp(int sceneViewIndex, SLMouseButton button, 
               int xpos, int ypos, SLKey modifier) 
{  
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseUp);
    e->svIndex = sceneViewIndex;
    e->button = button;
    e->x = xpos;
    e->y = ypos;
    e->modifier = modifier;
    SLApplication::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for double click events.
*/
void slDoubleClick(int sceneViewIndex, SLMouseButton button, 
                   int xpos, int ypos, SLKey modifier) 
{  
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseDoubleClick);
    e->svIndex = sceneViewIndex;
    e->button = button;
    e->x = xpos;
    e->y = ypos;
    e->modifier = modifier;
    SLApplication::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for long touches
*/
void slLongTouch(int sceneViewIndex, int xpos, int ypos) 
{  
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::LongTouch);
    e->svIndex = sceneViewIndex;
    e->x = xpos;
    e->y = ypos;
    SLApplication::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for the two finger touch down events of touchscreen 
devices.
*/
void slTouch2Down(int sceneViewIndex, int xpos1, int ypos1, int xpos2, int ypos2) 
{  
    SLTouchEvent* e = new SLTouchEvent(SLInputEvent::Touch2Down);
    e->svIndex = sceneViewIndex;
    e->x1 = xpos1;
    e->y1 = ypos1;
    e->x2 = xpos2;
    e->y2 = ypos2;

    SLApplication::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for the two finger move events of touchscreen devices. 
*/
void slTouch2Move(int sceneViewIndex, int xpos1, int ypos1, int xpos2, int ypos2) 
{  
    SLTouchEvent* e = new SLTouchEvent(SLInputEvent::Touch2Move);
    e->svIndex = sceneViewIndex;
    e->x1 = xpos1;
    e->y1 = ypos1;
    e->x2 = xpos2;
    e->y2 = ypos2;
    SLApplication::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for the two finger touch up events of touchscreen 
devices. 
*/
void slTouch2Up(int sceneViewIndex, int xpos1, int ypos1, int xpos2, int ypos2) 
{
    SLTouchEvent* e = new SLTouchEvent(SLInputEvent::Touch2Up);
    e->svIndex = sceneViewIndex;
    e->x1 = xpos1;
    e->y1 = ypos1;
    e->x2 = xpos2;
    e->y2 = ypos2;
    SLApplication::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse wheel events. 
*/
void slMouseWheel(int sceneViewIndex, int pos, SLKey modifier)
{  
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseWheel);
    e->svIndex = sceneViewIndex;
    e->y = pos;
    e->modifier = modifier;
    SLApplication::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for keyboard key press events. 
*/
void slKeyPress(int sceneViewIndex, SLKey key, SLKey modifier) 
{  
    SLKeyEvent* e = new SLKeyEvent(SLInputEvent::KeyDown);
    e->svIndex = sceneViewIndex;
    e->key = key;
    e->modifier = modifier;
    SLApplication::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for keyboard key release events. 
*/
void slKeyRelease(int sceneViewIndex, SLKey key, SLKey modifier) 
{  
    SLKeyEvent* e = new SLKeyEvent(SLInputEvent::KeyUp);
    e->svIndex = sceneViewIndex;
    e->key = key;
    e->modifier = modifier;
    SLApplication::inputManager.queueEvent(e);
}

//-----------------------------------------------------------------------------
/*! Global event handler for unicode character input.
*/
void slCharInput(int sceneViewIndex, unsigned int character)
{
    SLCharInputEvent* e = new SLCharInputEvent();
    e->svIndex = sceneViewIndex;
    e->character = character;
    SLApplication::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
bool slUsesRotation()
{
    if (SLApplication::scene)
        return SLApplication::devRot.isUsed();
    return false;
}
//-----------------------------------------------------------------------------
/*! Global event handler for device rotation change with angle & and axis.
*/
void slRotationQUAT(float quatX, float quatY, float quatZ, float quatW)
{
    SLApplication::devRot.onRotationQUAT(quatX, quatY, quatZ, quatW);
}
//-----------------------------------------------------------------------------
bool slUsesLocation()
{
    return SLApplication::devLoc.isUsed();
}
//-----------------------------------------------------------------------------
/*! Global event handler for device GPS location with longitude and latitude in
degrees and altitude in meters. This location uses the World Geodetic System
1984 (WGS 84). The accuracy in meters is a radius in which the location is with
a probability of 68% (2 sigma).
*/
void slLocationLLA(double latitudeDEG,
                   double longitudeDEG,
                   double altitudeM,
                   float  accuracyM)
{
    SLApplication::devLoc.onLocationLLA(latitudeDEG,
                                        longitudeDEG,
                                        altitudeM,
                                        accuracyM);
}
//-----------------------------------------------------------------------------
/*! Global function to retrieve a window title text generated by the scene
library. 
*/
string slGetWindowTitle(int sceneViewIndex) 
{  
    SLSceneView* sv = SLApplication::scene->sv(sceneViewIndex);
    return sv->windowTitle();
}
//-----------------------------------------------------------------------------
/*! Global function that returns the type of video camera wanted
*/
int slGetVideoType()
{
    return (int)SLApplication::scene->videoType();
}
//-----------------------------------------------------------------------------
/*! Global function that returns the size index offset of the requested video.
An index offset of 0 returns the default size of 640x480. If this size is not
available the median element of the available sizes array is returned.
An index of -n return the n-th smaller one. \n
An index of +n return the n-th bigger one. \n
*/
int slGetVideoSizeIndex()
{
    return SLCVCapture::requestedSizeIndex;
}
//-----------------------------------------------------------------------------
/*! Global function to grab the next frame with the OpenCV capture device. This
should be used by Android and iOS apps for grabbing the next video frame from
a video file.
*/
void slGrabVideoFileFrame()
{
    SLCVCapture::grabAndAdjustForSL();
}
//-----------------------------------------------------------------------------
/*! Global function to copy a new video image to the SLScene::_videoTexture.
An application can grab the live video image with OpenCV via slGrabCopyVideoImage
or with another OS dependent framework.
*/
void slCopyVideoImage(SLint width,
                      SLint height,
                      SLPixelFormat format,
                      SLuchar* data,
                      SLbool isContinuous)
{
    SLCVCapture::loadIntoLastFrame(width,
                                   height,
                                   format,
                                   data,
                                   isContinuous);
}
//-----------------------------------------------------------------------------
/*! Global function to copy a new video image in the YUV_420 format plane by
plane to the SLCVCapture::lastFrame. This should mainly used by mobile platforms
to efficiently copy the video frame to the SLCVCapture::lastFrame.
*/
void slCopyVideoYUVPlanes(int srcW, int srcH,
                          SLuchar* y, int ySize, int yPixStride, int yLineStride,
                          SLuchar* u, int uSize, int uPixStride, int uLineStride,
                          SLuchar* v, int vSize, int vPixStride, int vLineStride)
{
    SLCVCapture::copyYUVPlanes(srcW, srcH,
                               y, ySize, yPixStride, yLineStride,
                               u, uSize, uPixStride, uLineStride,
                               v, vSize, vPixStride, vLineStride);
}
//-----------------------------------------------------------------------------

