//#############################################################################
//  File:      SLApplication.h
//  Author:    Marcus Hudritsch
//  Date:      February 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLAPPLICATION_H
#define SLAPPLICATION_H

#include <CVTypes.h>
#include <SLDeviceLocation.h>
#include <SLDeviceRotation.h>
#include <SLInputManager.h>
#include <atomic>
#include <mutex>
#include <map>

using namespace std;

class SLScene;
class SLSceneView;
class SLGLImGui;
class SLProjectScene;
class CVCalibrationEstimator;
//-----------------------------------------------------------------------------
//! Top level class for an SLProject application.
/*!      
 The SLApplication holds static instances of top-level items such as the scene
 pointer, the camera calibration objects and the device rotation and location
 information. The static function createAppAndScene is called by the C-interface
 functions slCreateAppAndScene and the function deleteAppAndScene by slTerminate.
 At the moment only one scene can be open at the time.
 <br>
 SLApplication holds two static video camera calibrations, one for a main camera
 (mainCam) and one for the selfie camera on mobile devices (scndCam).
 The pointer activeCamera points to the active one.
*/
class SLApplication
{
public:
    static void   createAppAndScene(SLstring appName,
                                    void*    onSceneLoadCallback);
    static void   deleteAppAndScene();
    static void   handleParallelJob();
    static void   jobProgressMsg(string msg);
    static void   jobProgressNum(int num) { _jobProgressNum = num; }
    static void   jobProgressMax(int max) { _jobProgressMax = max; }
    static string jobProgressMsg();
    static int    jobProgressNum() { return _jobProgressNum; }
    static int    jobProgressMax() { return _jobProgressMax; }

    static SLProjectScene*      scene;        //!< scene pointer
    static vector<SLSceneView*> sceneViews;   //!< vector of sceneview pointers
    static SLGLImGui*           gui;          //!< gui pointer
    static SLInputManager       inputManager; //!< Input events manager
    static SLDeviceRotation     devRot;       //!< Mobile device rotation from IMU
    static SLDeviceLocation     devLoc;       //!< Mobile device location from GPS

    static SLstring name;          //!< Application name
    static SLstring appTag;        //!< Tag string used in logging
    static SLstring version;       //!< SLProject version string
    static SLstring configuration; //!< Debug or Release configuration
    static SLstring gitBranch;     //!< Current GIT branch
    static SLstring gitCommit;     //!< Current GIT commit short hash id
    static SLstring gitDate;       //!< Current GIT commit date
    static SLstring exePath;       //!< executable root path
    static SLstring configPath;    //!< Default path for calibration files
    static SLstring externalPath;  //!< Default path for external file storage
    static SLstring dataPath;      //!< Path to data directory (it is set platform dependent)
    static SLstring shaderPath;
    static SLstring modelPath;
    static SLstring texturePath;
    static SLstring fontPath;

    static SLSceneID sceneID; //!< ID of last loaded scene

    static map<string, string>         deviceParameter;    //!< Generic device parameter
    static deque<function<void(void)>> jobsToBeThreaded;   //!< Queue of functions to be executed in a thread
    static deque<function<void(void)>> jobsToFollowInMain; //!< Queue of function to follow in the main thread
    static atomic<bool>                jobIsRunning;       //!< True if a parallel job is running

    static CVCalibrationEstimatorParams calibrationEstimatorParams;
    static CVCalibrationEstimator*      calibrationEstimator;
    static SLstring                     calibIniPath;  //!< That's where data/calibrations folder is located
    static SLstring                     calibFilePath; //!< That's where calibrations are stored and loaded from

    static const string CALIB_FTP_HOST; //!< ftp host for calibration up and download
    static const string CALIB_FTP_USER; //!< ftp login user for calibration up and download
    static const string CALIB_FTP_PWD;  //!< ftp login pwd for calibration up and download
    static const string CALIB_FTP_DIR;  //!< ftp directory for calibration up and download

private:
    static string      _jobProgressMsg; //!< Text message to show during progress
    static atomic<int> _jobProgressNum; //!< Integer value to show progress
    static atomic<int> _jobProgressMax; //!< Max. integer progress value
    static mutex       _jobMutex;       //!< Mutex to protect parallel access
};
//-----------------------------------------------------------------------------
#endif
