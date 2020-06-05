//#############################################################################
//  File:      SLApplication.h
//  Author:    Marcus Hudritsch
//  Date:      Februar 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <SL.h>
#include <SLApplication.h>
#include <SLSceneView.h>
#include <SLProjectScene.h>
#include <SLGLImGui.h>
#include <utility>
#include <GlobalTimer.h>
#include <SLGLProgramManager.h>
#include <SLOptix.h>

//-----------------------------------------------------------------------------
//! Global static objects
SLInputManager       SLApplication::inputManager;
SLProjectScene*      SLApplication::scene = nullptr;
vector<SLSceneView*> SLApplication::sceneViews;
SLGLImGui*           SLApplication::gui = nullptr;
SLDeviceRotation     SLApplication::devRot;
SLDeviceLocation     SLApplication::devLoc;
SLstring             SLApplication::name    = "SLProjectApp";
SLstring             SLApplication::appTag  = "SLProject";
SLstring             SLApplication::version = "3.0.000";
#ifdef _DEBUG
SLstring SLApplication::configuration = "Debug";
#else
SLstring SLApplication::configuration = "Release";
#endif
SLstring            SLApplication::gitBranch = SL_GIT_BRANCH;
SLstring            SLApplication::gitCommit = SL_GIT_COMMIT;
SLstring            SLApplication::gitDate   = SL_GIT_DATE;
map<string, string> SLApplication::deviceParameter;

CVCalibrationEstimatorParams SLApplication::calibrationEstimatorParams;
CVCalibrationEstimator*      SLApplication::calibrationEstimator = nullptr;
SLstring                     SLApplication::calibIniPath;
SLstring                     SLApplication::calibFilePath;

//! SLApplication::configPath is overwritten in slCreateAppAndScene.
SLstring SLApplication::exePath      = SLstring(SL_PROJECT_ROOT) + "/";
SLstring SLApplication::configPath   = SLstring(SL_PROJECT_ROOT) + "/data/config/";
SLstring SLApplication::externalPath = SLstring(SL_PROJECT_ROOT) + "/data/config/";
SLstring SLApplication::dataPath;
SLstring SLApplication::shaderPath;
SLstring SLApplication::modelPath;
SLstring SLApplication::texturePath;
SLstring SLApplication::fontPath;
SLstring SLApplication::videoPath;

SLSceneID                   SLApplication::sceneID = SID_Empty;
deque<function<void(void)>> SLApplication::jobsToBeThreaded;
deque<function<void(void)>> SLApplication::jobsToFollowInMain;
atomic<bool>                SLApplication::jobIsRunning(false);
string                      SLApplication::_jobProgressMsg = "";
atomic<int>                 SLApplication::_jobProgressNum(0);
atomic<int>                 SLApplication::_jobProgressMax(0);
mutex                       SLApplication::_jobMutex;

const string SLApplication::CALIB_FTP_HOST  = "pallas.bfh.ch:21";
const string SLApplication::CALIB_FTP_USER  = "upload";
const string SLApplication::CALIB_FTP_PWD   = "FaAdbD3F2a";
const string SLApplication::CALIB_FTP_DIR   = "calibrations";
const string SLApplication::PROFILE_FTP_DIR = "profiles";

//-----------------------------------------------------------------------------
//! Application and Scene creation function
/*! Writes and inits the static application information and create the single
instance of the scene. Gets called by the C-interface function slCreateAppAndScene.
<br>
<br>
See examples usages in:
  - app-Demo-SLProject/GLFW:    AppDemoMainGLFW.cpp   in function main()
  - app-Demo-SLProject/android: AppDemoAndroidJNI.cpp in Java_ch_fhnw_comgr_GLES3Lib_onInit()
  - app-Demo-SLProject/iOS:     ViewController.m      in viewDidLoad()
<br>
/param applicationName The apps name
/param onSceneLoadCallback C Callback function as void* pointer for the scene creation.
*/
void SLApplication::createAppAndScene(SLstring appName,
                                      void*    onSceneLoadCallback)
{
    assert(SLApplication::scene == nullptr &&
           "You can create only one SLApplication");

    name = std::move(appName);
    SLGLProgramManager::init(dataPath + "shaders/");
    scene = new SLProjectScene(name, (cbOnSceneLoad)onSceneLoadCallback);
    scene->initOculus(dataPath + "shaders/");
    GlobalTimer::timerStart();

#ifdef SL_HAS_OPTIX
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
void SLApplication::deleteAppAndScene()
{
    assert(SLApplication::scene != nullptr &&
           "You can delete an  only once");

    for (auto* sv : sceneViews)
        delete sv;
    sceneViews.clear();

    delete scene;
    scene = nullptr;
    if (gui)
    {
        delete gui;
        gui = nullptr;
    }

    //delete default stuff:
    SLGLProgramManager::deletePrograms();
    SLMaterialDefaultGray::deleteInstance();
    SLMaterialDiffuseAttribute::deleteInstance();
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
void SLApplication::handleParallelJob()
{
    // Execute queued jobs in a parallel thread
    if (!SLApplication::jobIsRunning &&
        !SLApplication::jobsToBeThreaded.empty())
    {
        function<void(void)> job = SLApplication::jobsToBeThreaded.front();
        thread               jobThread(job);
        SLApplication::jobIsRunning = true;
        SLApplication::jobsToBeThreaded.pop_front();
        jobThread.detach();
    }

    // Execute the jobs to follow in the this (the main) thread
    if (!SLApplication::jobIsRunning &&
        SLApplication::jobsToBeThreaded.empty() &&
        !SLApplication::jobsToFollowInMain.empty())
    {
        for (const auto& jobToFollow : SLApplication::jobsToFollowInMain)
            jobToFollow();
        SLApplication::jobsToFollowInMain.clear();
    }
}
//-----------------------------------------------------------------------------
//! Threadsafe setter of the progress message
void SLApplication::jobProgressMsg(string msg)
{
    SLApplication::_jobMutex.lock();
    SLApplication::_jobProgressMsg = std::move(msg);
    SLApplication::_jobMutex.unlock();
}
//-----------------------------------------------------------------------------
//! Threadsafe getter of the progress message
string SLApplication::jobProgressMsg()
{
    lock_guard<mutex> guard(_jobMutex);
    return _jobProgressMsg;
}
//-----------------------------------------------------------------------------
