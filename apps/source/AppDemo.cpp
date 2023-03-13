//#############################################################################
//  File:      AppDemo.h
//  Date:      Februar 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SL.h>
#include <AppDemo.h>
#include <SLAssetManager.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLGLImGui.h>
#include <utility>
#include <GlobalTimer.h>
#include <SLGLProgramManager.h>
#include <SLOptix.h>

//-----------------------------------------------------------------------------
//! Global static objects
SLInputManager       AppDemo::inputManager;
SLAssetManager*      AppDemo::assetManager = nullptr;
SLScene*             AppDemo::scene        = nullptr;
vector<SLSceneView*> AppDemo::sceneViews;
SLGLImGui*           AppDemo::gui = nullptr;
SLDeviceRotation     AppDemo::devRot;
SLDeviceLocation     AppDemo::devLoc;
SLstring             AppDemo::name    = "SLProjectApp";
SLstring             AppDemo::appTag  = "SLProject";
SLstring             AppDemo::version = "3.4.000";
#ifdef _DEBUG
SLstring AppDemo::configuration = "Debug";
#else
SLstring AppDemo::configuration = "Release";
#endif
SLstring            AppDemo::gitBranch = SL_GIT_BRANCH;
SLstring            AppDemo::gitCommit = SL_GIT_COMMIT;
SLstring            AppDemo::gitDate   = SL_GIT_DATE;
map<string, string> AppDemo::deviceParameter;

CVCalibrationEstimatorParams AppDemo::calibrationEstimatorParams;
CVCalibrationEstimator*      AppDemo::calibrationEstimator = nullptr;
SLstring                     AppDemo::calibIniPath;
SLstring                     AppDemo::calibFilePath;

SLstring AppDemo::exePath;
SLstring AppDemo::configPath;
SLstring AppDemo::externalPath;
SLstring AppDemo::dataPath;
SLstring AppDemo::shaderPath;
SLstring AppDemo::modelPath;
SLstring AppDemo::texturePath;
SLstring AppDemo::fontPath;
SLstring AppDemo::videoPath;

SLSceneID                   AppDemo::sceneID = SID_Empty;
deque<function<void(void)>> AppDemo::jobsToBeThreaded;
deque<function<void(void)>> AppDemo::jobsToFollowInMain;
atomic<bool>                AppDemo::jobIsRunning(false);
string                      AppDemo::_jobProgressMsg = "";
atomic<int>                 AppDemo::_jobProgressNum(0);
atomic<int>                 AppDemo::_jobProgressMax(0);
mutex                       AppDemo::_jobMutex;

const string AppDemo::CALIB_FTP_HOST  = "pallas.ti.bfh.ch:21";
const string AppDemo::CALIB_FTP_USER  = "upload";
const string AppDemo::CALIB_FTP_PWD   = "FaAdbD3F2a";
const string AppDemo::CALIB_FTP_DIR   = "calibrations";
const string AppDemo::PROFILE_FTP_DIR = "profiles";

//-----------------------------------------------------------------------------
//! Application and Scene creation function
/*! Writes and inits the static application information and create the single
instance of the scene. Gets called by the C-interface function slCreateAppAndScene.
<br>
<br>
See examples usages in:
  - app_demo_slproject/glfw:    AppDemoMainGLFW.cpp   in function main()
  - app_demo_slproject/android: AppDemoAndroidJNI.cpp in Java_ch_fhnw_comgr_GLES3Lib_onInit()
  - app_demo_slproject/ios:     ViewController.m      in viewDidLoad()
<br>
/param applicationName The apps name
/param onSceneLoadCallback C Callback function as void* pointer for the scene creation.
*/
void AppDemo::createAppAndScene(SLstring appName,
                                void*    onSceneLoadCallback)
{
    assert(AppDemo::scene == nullptr &&
           "You can create only one AppDemo");

    name = std::move(appName);
    SLGLProgramManager::init(dataPath + "shaders/", configPath);
    assetManager = new SLAssetManager(AppDemo::fontPath, true);
    scene        = new SLScene(name, (cbOnSceneLoad)onSceneLoadCallback);
    scene->initOculus(dataPath + "shaders/");
    GlobalTimer::timerStart();

#ifdef SL_HAS_OPTIX
    SLOptix::exePath = AppDemo::exePath;
    SLOptix::createStreamAndContext();
#endif
}
//-----------------------------------------------------------------------------
//! Calls the destructor of the single scene instance.
/*! Destroys all data by calling the destructor of the single scene instance.
All other date gets destroyed from there. This function gets called by the
SLProject C-Interface function slTerminate that should be called at the end of
any SLProject application.
*/
void AppDemo::deleteAppAndScene()
{
    // The WebGL context is apparently already destroyed when we call this function
#ifndef SL_EMSCRIPTEN
    assert(AppDemo::scene != nullptr &&
           "You can delete a scene only once");

    for (auto* sv : sceneViews)
        delete sv;
    sceneViews.clear();

    delete scene;
    scene = nullptr;

    delete assetManager;
    assetManager = nullptr;
#endif

    if (gui)
    {
        delete gui;
        gui = nullptr;
    }

    // The WebGL context is apparently already destroyed when we call this function
#ifndef SL_EMSCRIPTEN
    // delete default stuff:
    SLGLProgramManager::deletePrograms();
    SLMaterialDefaultGray::deleteInstance();
#endif
}
//-----------------------------------------------------------------------------
//! Starts parallel job if one is queued.
/*!
Parallel executed job can be queued in jobsToBeThreaded. Only functions are
allowed that do not call any OpenGL functions. So no scenegraph changes are
allowed because they involve mostly OpenGL state and context changes.
Only one parallel job is executed at once parallel to the main rendering thread.
The function in jobsToFollowInMain will be executed in the main tread after the
parallel are finished.<br>
The handleParallelJob function gets called in slUpdateAndPaint before a new
frame gets started. See an example parallel job definition in AppDemoGui.
If a parallel job is running the jobProgressMsg can be shown during execution.
If jobProgressMax is 0 the jobProgressNum value can be shown an number.
If jobProgressMax is not 0 the fraction of jobProgressNum/jobProgressMax can
be shown within a progress bar. See the example in AppDemoGui::build.
*/
void AppDemo::handleParallelJob()
{
    // Execute queued jobs in a parallel thread
    if (!AppDemo::jobIsRunning &&
        !AppDemo::jobsToBeThreaded.empty())
    {
        function<void(void)> job = AppDemo::jobsToBeThreaded.front();
        thread               jobThread(job);
        AppDemo::jobIsRunning = true;
        AppDemo::jobsToBeThreaded.pop_front();
        jobThread.detach();
    }

    // Execute the jobs to follow in the this (the main) thread
    if (!AppDemo::jobIsRunning &&
        AppDemo::jobsToBeThreaded.empty() &&
        !AppDemo::jobsToFollowInMain.empty())
    {
        for (const auto& jobToFollow : AppDemo::jobsToFollowInMain)
            jobToFollow();
        AppDemo::jobsToFollowInMain.clear();
    }
}
//-----------------------------------------------------------------------------
//! Thread-safe setter of the progress message
void AppDemo::jobProgressMsg(string msg)
{
    AppDemo::_jobMutex.lock();
    AppDemo::_jobProgressMsg = std::move(msg);
    AppDemo::_jobMutex.unlock();
}
//-----------------------------------------------------------------------------
//! Thread-safe getter of the progress message
string AppDemo::jobProgressMsg()
{
    lock_guard<mutex> guard(_jobMutex);
    return _jobProgressMsg;
}
//-----------------------------------------------------------------------------
