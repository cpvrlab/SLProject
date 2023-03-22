//#############################################################################
//  File:      sl/SLInterface.cpp
//  Purpose:   Implementation of the main Scene Library C-Interface.
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLInterface.h>
#include <AppDemo.h>
#include <SLAssimpImporter.h>
#include <SLInputManager.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLGLImGui.h>
#include <Profiler.h>
#include <ZipUtils.h>

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
  - app_demo_slproject/glfw:    AppDemoMainGLFW.cpp in function main()
  - app-Demo-SLProject/android: native-lib.cpp      in Java_ch_fhnw_comgr_GLES3Lib_onInit()
  - app_demo_slproject/ios:     ViewController.m    in viewDidLoad()
*/
void slCreateAppAndScene(SLVstring&      cmdLineArgs,
                         const SLstring& dataPath,
                         const SLstring& shaderPath,
                         const SLstring& modelPath,
                         const SLstring& texturePath,
                         const SLstring& fontPath,
                         const SLstring& videoPath,
                         const SLstring& configPath,
                         const SLstring& applicationName,
                         void*           onSceneLoadCallback)
{
    assert(AppDemo::scene == nullptr && "SLScene is already created!");

    // For more info on PROFILING read Utils/lib-utils/source/Profiler.h
#if PROFILING
    if (Utils::dirExists(AppDemo::externalPath))
    {
        SLstring computerInfo = Utils::ComputerInfos::get();
        SLstring profileFile  = AppDemo::externalPath + "Profile_" + computerInfo + ".slt";
        BEGIN_PROFILING_SESSION(profileFile);
        PROFILE_THREAD("Main Thread");
    }
#endif

    // Default paths for all loaded resources
    SLstring exe         = Utils::unifySlashes(cmdLineArgs.size() ? cmdLineArgs[0] : "", false);
    AppDemo::exePath     = Utils::getDirName(exe);
    AppDemo::dataPath    = Utils::unifySlashes(dataPath);
    AppDemo::shaderPath  = shaderPath;
    AppDemo::modelPath   = modelPath;
    AppDemo::texturePath = texturePath;
    AppDemo::fontPath    = fontPath;
    AppDemo::videoPath   = videoPath;
    AppDemo::configPath  = configPath;

    SLGLState* stateGL = SLGLState::instance();

    SL_LOG("Path to exe      : %s", AppDemo::exePath.c_str());
    SL_LOG("Path to Models   : %s", modelPath.c_str());
    SL_LOG("Path to Shaders  : %s", shaderPath.c_str());
    SL_LOG("Path to Textures : %s", texturePath.c_str());
    SL_LOG("Path to Fonts    : %s", fontPath.c_str());
    SL_LOG("Path to Config.  : %s", configPath.c_str());
    SL_LOG("Path to Documents: %s", AppDemo::externalPath.c_str());
    SL_LOG("OpenCV Version   : %d.%d.%d", CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_VERSION_REVISION);
    SL_LOG("OpenCV has OpenCL: %s", cv::ocl::haveOpenCL() ? "yes" : "no");
    SL_LOG("OpenGL Ver. Str. : %s", stateGL->glVersion().c_str());
    SL_LOG("OpenGL Ver. No.  : %s", stateGL->glVersionNO().c_str());
    SL_LOG("OpenGL Vendor    : %s", stateGL->glVendor().c_str());
    SL_LOG("OpenGL Renderer  : %s", stateGL->glRenderer().c_str());
    SL_LOG("OpenGL GLSL Ver. : %s (%s) ", stateGL->glSLVersion().c_str(), stateGL->getSLVersionNO().c_str());
    SL_LOG("------------------------------------------------------------------");

    AppDemo::createAppAndScene(applicationName, onSceneLoadCallback);
}
//-----------------------------------------------------------------------------
/*! Global creation function for a SLSceneview instance returning the index of
the sceneview. It creates the new SLSceneView instance by calling the callback
function slNewSceneView. If you have a custom SLSceneView inherited class you
have to provide a similar function and pass it function pointer to
slCreateSceneView. You can create multiple sceneview per application.<br>
<br>
See examples usages in:
  - app_demo_slproject/glfw:    AppDemoMainGLFW.cpp   in function main()
  - app-Demo-SLProject/android: AppDemoAndroidJNI.cpp in Java_ch_fhnw_comgr_GLES3Lib_onInit()
  - app_demo_slproject/ios:     ViewController.m      in viewDidLoad()
*/
SLint slCreateSceneView(SLAssetManager* am,
                        SLScene*        scene,
                        int             screenWidth,
                        int             screenHeight,
                        int             dotsPerInch,
                        SLSceneID       initScene,
                        void*           onWndUpdateCallback,
                        void*           onSelectNodeMeshCallback,
                        void*           onNewSceneViewCallback,
                        void*           onImGuiBuild,
                        void*           onImGuiLoadConfig,
                        void*           onImGuiSaveConfig)
{
    assert(scene && "No valid scene!");

    // Use our own sceneview creator callback or the passed one.
    cbOnNewSceneView newSVCallback;
    if (onNewSceneViewCallback == nullptr)
        newSVCallback = &slNewSceneView;
    else
        newSVCallback = (cbOnNewSceneView)onNewSceneViewCallback;

    // Create the sceneview & get the pointer with the sceneview index
    SLSceneView* sv = newSVCallback(scene, dotsPerInch, AppDemo::inputManager);

    // maintain multiple scene views in AppDemo
    AppDemo::sceneViews.push_back(sv);

    AppDemo::gui = new SLGLImGui((cbOnImGuiBuild)onImGuiBuild,
                                 (cbOnImGuiLoadConfig)onImGuiLoadConfig,
                                 (cbOnImGuiSaveConfig)onImGuiSaveConfig,
                                 dotsPerInch,
                                 AppDemo::fontPath);

    sv->init("SceneView",
             screenWidth,
             screenHeight,
             onWndUpdateCallback,
             onSelectNodeMeshCallback,
             AppDemo::gui,
             AppDemo::configPath);

    // Set active sceneview and load scene. This is done for the first sceneview
    if (!scene->root3D())
    {
        if (AppDemo::sceneID == SID_Empty)
            scene->onLoad(am, AppDemo::scene, sv, initScene);
        else
            scene->onLoad(am, scene, sv, AppDemo::sceneID);
    }
    else
        sv->onInitialize();

    // return the identifier index
    return (SLint)AppDemo::sceneViews.size() - 1;
}
//-----------------------------------------------------------------------------
/*! Global sceneview construction function returning the index of the created
sceneview instance. If you have a custom SLSceneView inherited class you
have to provide a similar function and pass it function pointer to
slCreateSceneView.
*/
SLSceneView* slNewSceneView(SLScene*        s,
                            int             dotsPerInch,
                            SLInputManager& inputManager)
{
    return new SLSceneView(s, dotsPerInch, inputManager);
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
    SL_LOG("Begin of Terminate");

    // Deletes all remaining sceneviews the current scene instance
    AppDemo::deleteAppAndScene();

    // For more info on PROFILING read Utils/lib-utils/source/Profiler.h
#if PROFILING
    SLstring filePathName = PROFILER_TRACE_FILE_PATH;

    SL_LOG("Before END_PROFILING_SESSION");
    END_PROFILING_SESSION();
    SL_LOG("After END_PROFILING_SESSION");

    Profiler::instance().endSession();

    if (Utils::fileExists(filePathName))
    {
        SL_LOG("Profile written : %s", filePathName.c_str());
        /*
        //ZipUtils::zip("/Users/hudrima1/Library/Application Support/SLProject/DEVELOPMENT-map_20200529-154142_avenches_aamphitheater_FAST_ORBS_2000.json");

        SLstring errorMsg;
        SLstring path = Utils::getPath(filePathName);
        SLstring file = Utils::getFileName(filePathName);

        SL_LOG("Profile Uploading ...");

        if (FtpUtils::uploadFile(path,
                                 file,
                                 AppDemo::CALIB_FTP_HOST,
                                 AppDemo::CALIB_FTP_USER,
                                 AppDemo::CALIB_FTP_PWD,
                                 AppDemo::PROFILE_FTP_DIR,
                                 errorMsg))
        {
            SL_LOG("Uploaded Profile: %s", filePathName.c_str());
        } else
            SL_LOG(errorMsg.c_str());
        */
    }
    else
        SL_LOG("No Profile written: %s", filePathName.c_str());
#else
    SL_LOG("No Profiling");
#endif

    SL_LOG("End of Terminate");
    SL_LOG("------------------------------------------------------------------");
}
//-----------------------------------------------------------------------------
/*!
 * Updates the parallel running job an allowes the update of a progress bar.
 * @return Returns true if parallel jobs are still running.
 */
bool slUpdateParallelJob()
{
    AppDemo::handleParallelJob();
    return AppDemo::jobIsRunning;
}
//-----------------------------------------------------------------------------
/*!
 * Draws all scene views
 * @return return true if another repaint is needed.
 */
bool slPaintAllViews()
{
    bool needUpdate = false;

    for (auto sv : AppDemo::sceneViews)
    {
        // Save previous frame as image
        if (sv->screenCaptureIsRequested())
        {
            SLstring path = AppDemo::externalPath + "screenshots/";
            Utils::makeDirRecurse(path);
            SLstring filename     = "Screenshot_" + Utils::getDateTime2String() + ".png";
            SLstring pathFilename = path + filename;
            sv->saveFrameBufferAsImage(pathFilename);
        }

        if (sv->onPaint() && !needUpdate)
            needUpdate = true;
    }

    return needUpdate;
}
//-----------------------------------------------------------------------------
/*! Global resize function that must be called whenever the OpenGL frame
changes it's size.
*/
void slResize(int sceneViewIndex, int width, int height)
{
    SLResizeEvent* e = new SLResizeEvent;
    e->svIndex       = sceneViewIndex;
    e->width         = width;
    e->height        = height;
    AppDemo::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse button down events.
 */
void slMouseDown(int           sceneViewIndex,
                 SLMouseButton button,
                 int           xpos,
                 int           ypos,
                 SLKey         modifier)
{
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseDown);
    e->svIndex      = sceneViewIndex;
    e->button       = button;
    e->x            = xpos;
    e->y            = ypos;
    e->modifier     = modifier;
    AppDemo::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse move events.
 */
void slMouseMove(int sceneViewIndex,
                 int x,
                 int y)
{
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseMove);
    e->svIndex      = sceneViewIndex;
    e->x            = x;
    e->y            = y;
    AppDemo::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse button up events.
 */
void slMouseUp(int           sceneViewIndex,
               SLMouseButton button,
               int           xpos,
               int           ypos,
               SLKey         modifier)
{
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseUp);
    e->svIndex      = sceneViewIndex;
    e->button       = button;
    e->x            = xpos;
    e->y            = ypos;
    e->modifier     = modifier;
    AppDemo::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for double click events.
 */
void slDoubleClick(int           sceneViewIndex,
                   SLMouseButton button,
                   int           xpos,
                   int           ypos,
                   SLKey         modifier)
{
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseDoubleClick);
    e->svIndex      = sceneViewIndex;
    e->button       = button;
    e->x            = xpos;
    e->y            = ypos;
    e->modifier     = modifier;
    AppDemo::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for the two finger touch down events of touchscreen
devices.
*/
void slTouch2Down(int sceneViewIndex,
                  int xpos1,
                  int ypos1,
                  int xpos2,
                  int ypos2)
{
    SLTouchEvent* e = new SLTouchEvent(SLInputEvent::Touch2Down);
    e->svIndex      = sceneViewIndex;
    e->x1           = xpos1;
    e->y1           = ypos1;
    e->x2           = xpos2;
    e->y2           = ypos2;

    AppDemo::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for the two finger move events of touchscreen devices.
 */
void slTouch2Move(int sceneViewIndex,
                  int xpos1,
                  int ypos1,
                  int xpos2,
                  int ypos2)
{
    SLTouchEvent* e = new SLTouchEvent(SLInputEvent::Touch2Move);
    e->svIndex      = sceneViewIndex;
    e->x1           = xpos1;
    e->y1           = ypos1;
    e->x2           = xpos2;
    e->y2           = ypos2;
    AppDemo::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for the two finger touch up events of touchscreen
devices.
*/
void slTouch2Up(int sceneViewIndex,
                int xpos1,
                int ypos1,
                int xpos2,
                int ypos2)
{
    SLTouchEvent* e = new SLTouchEvent(SLInputEvent::Touch2Up);
    e->svIndex      = sceneViewIndex;
    e->x1           = xpos1;
    e->y1           = ypos1;
    e->x2           = xpos2;
    e->y2           = ypos2;
    AppDemo::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for mouse wheel events.
 */
void slMouseWheel(int   sceneViewIndex,
                  int   pos,
                  SLKey modifier)
{
    SLMouseEvent* e = new SLMouseEvent(SLInputEvent::MouseWheel);
    e->svIndex      = sceneViewIndex;
    e->y            = pos;
    e->modifier     = modifier;
    AppDemo::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for keyboard key press events.
 */
void slKeyPress(int   sceneViewIndex,
                SLKey key,
                SLKey modifier)
{
    SLKeyEvent* e = new SLKeyEvent(SLInputEvent::KeyDown);
    e->svIndex    = sceneViewIndex;
    e->key        = key;
    e->modifier   = modifier;
    AppDemo::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
/*! Global event handler for keyboard key release events.
 */
void slKeyRelease(int   sceneViewIndex,
                  SLKey key,
                  SLKey modifier)
{
    SLKeyEvent* e = new SLKeyEvent(SLInputEvent::KeyUp);
    e->svIndex    = sceneViewIndex;
    e->key        = key;
    e->modifier   = modifier;
    AppDemo::inputManager.queueEvent(e);
}

//-----------------------------------------------------------------------------
/*! Global event handler for unicode character input.
 */
void slCharInput(int          sceneViewIndex,
                 unsigned int character)
{
    SLCharInputEvent* e = new SLCharInputEvent();
    e->svIndex          = sceneViewIndex;
    e->character        = character;
    AppDemo::inputManager.queueEvent(e);
}
//-----------------------------------------------------------------------------
bool slUsesRotation()
{
    if (AppDemo::scene)
        return AppDemo::devRot.isUsed();
    return false;
}
//-----------------------------------------------------------------------------
/*! Global event handler for device rotation change with angle & and axis.
 */
void slRotationQUAT(float quatX,
                    float quatY,
                    float quatZ,
                    float quatW)
{
    AppDemo::devRot.onRotationQUAT(quatX, quatY, quatZ, quatW);
}
//-----------------------------------------------------------------------------
bool slUsesLocation()
{
    return AppDemo::devLoc.isUsed();
}
//-----------------------------------------------------------------------------
/*! Global event handler for device GPS location with longitude and latitude in
degrees and altitude in meters. This location uses the World Geodetic System
1984 (WGS 84). The accuracy in meters is a radius in which the location is with
a probability of 68% (2 sigma).
*/
void slLocationLatLonAlt(double latitudeDEG,
                         double longitudeDEG,
                         double altitudeM,
                         float  accuracyM)
{
    AppDemo::devLoc.onLocationLatLonAlt(latitudeDEG,
                                        longitudeDEG,
                                        altitudeM,
                                        accuracyM);
}
//-----------------------------------------------------------------------------
//! Global function to retrieve a window title generated by the scene library.
string slGetWindowTitle(int sceneViewIndex)
{
    SLSceneView* sv = AppDemo::sceneViews[(SLuint)sceneViewIndex];
    return sv->windowTitle();
}
//-----------------------------------------------------------------------------
// Get available external directories and inform slproject about them
void slSetupExternalDir(const SLstring& externalPath)
{
    if (Utils::dirExists(externalPath))
    {
        SL_LOG("Ext. directory   : %s", externalPath.c_str());
        AppDemo::externalPath = Utils::trimRightString(externalPath, "/") + "/";
    }
    else
    {
        SL_LOG("ERROR: external directory does not exists: %s", externalPath.c_str());
    }
}
//-----------------------------------------------------------------------------
//! Adds a value to the applications device parameter map
void slSetDeviceParameter(const SLstring& parameter,
                          SLstring        value)
{
    AppDemo::deviceParameter[parameter] = std::move(value);
}
//-----------------------------------------------------------------------------
