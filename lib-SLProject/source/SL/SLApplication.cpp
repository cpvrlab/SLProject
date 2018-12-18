//#############################################################################
//  File:      SLApplication.h
//  Author:    Marcus Hudritsch
//  Date:      Februar 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SLApplication.h>
#include <SLCVCapture.h>
#include <SLCVTracked.h>
#include <SLCVTrackedAruco.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
//! Global static objects
SLInputManager   SLApplication::inputManager;
SLScene*         SLApplication::scene       = nullptr;
SLCVCalibration* SLApplication::activeCalib = nullptr;
SLCVCalibration  SLApplication::calibMainCam;
SLCVCalibration  SLApplication::calibScndCam;
SLDeviceRotation SLApplication::devRot;
SLDeviceLocation SLApplication::devLoc;
SLstring         SLApplication::name      = "SLProjectApp";
SLstring         SLApplication::version   = "2.3.001";
SLstring         SLApplication::gitBranch = SL_GIT_BRANCH;
SLstring         SLApplication::gitCommit = SL_GIT_COMMIT;
SLstring         SLApplication::gitDate   = SL_GIT_DATE;
SLint            SLApplication::dpi       = 0;
//! SLApplication::configPath is overwritten in slCreateAppAndScene.
SLstring  SLApplication::configPath = SLstring(SL_PROJECT_ROOT) + "/data/config/";
SLSceneID SLApplication::sceneID    = SID_Empty;
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

    name = appName;

    scene = new SLScene(name, (cbOnSceneLoad)onSceneLoadCallback);

// load opencv camera calibration for main and secondary camera
#if defined(SL_USES_CVCAPTURE)
    calibMainCam.load("cam_calibration_main.xml", true, false);
    calibMainCam.loadCalibParams();
    activeCalib                     = &calibMainCam;
    SLCVCapture::hasSecondaryCamera = false;
#else
    calibMainCam.load("cam_calibration_main.xml", false, false);
    calibMainCam.loadCalibParams();
    calibScndCam.load("cam_calibration_scnd.xml", true, false);
    calibScndCam.loadCalibParams();
    activeCalib                     = &calibMainCam;
    SLCVCapture::hasSecondaryCamera = true;
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

    if (scene)
        delete scene;

    scene = nullptr;
}
//-----------------------------------------------------------------------------
