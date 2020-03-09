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
#include <HighResTimer.h>
#include <atomic>
#include <mutex>
#include <map>

using namespace std;

class SLScene;
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
    static void     createAppAndScene(SLstring appName,
                                      void*    onSceneLoadCallback);
    static void     deleteAppAndScene();
    static SLstring getComputerInfos();
    static void     handleParallelJob();
    static void     jobProgressMsg(string msg);
    static void     jobProgressNum(int num) { _jobProgressNum = num; }
    static void     jobProgressMax(int max) { _jobProgressMax = max; }
    static string   jobProgressMsg();
    static int      jobProgressNum() { return _jobProgressNum; }
    static int      jobProgressMax() { return _jobProgressMax; }
    static SLfloat  dpmm() { return (float)dpi / 25.4f; } //!< return dots per mm
    static void     timerStart() { _timer.start(); }
    static SLfloat  timeS() { return _timer.elapsedTimeInSec(); }
    static SLfloat  timeMS() { return _timer.elapsedTimeInMilliSec(); }

    static SLScene*         scene;        //!< scene pointer
    static SLInputManager   inputManager; //!< Input events manager
    static SLDeviceRotation devRot;       //!< Mobile device rotation from IMU
    static SLDeviceLocation devLoc;       //!< Mobile device location from GPS

    static SLstring  name;          //!< Applcation name
    static SLstring  version;       //!< SLProject version string
    static SLstring  configuration; //!< Debug or Release configuration
    static SLstring  computerUser;  //!< Computer Name (= env-var USER)
    static SLstring  computerName;  //!< Computer Name (= env-var HOSTNAME)
    static SLstring  computerBrand; //!< Computer brand name
    static SLstring  computerModel; //!< Computer model name
    static SLstring  computerOS;    //!< Computer OS name
    static SLstring  computerOSVer; //!< Computer OS version
    static SLstring  computerArch;  //!< Computer Architecture
    static SLstring  computerID;    //!< Computer identification string
    static SLstring  gitBranch;     //!< Current GIT branch
    static SLstring  gitCommit;     //!< Current GIT commit short hash id
    static SLstring  gitDate;       //!< Current GIT commit date
    static SLint     dpi;           //!< Current UI dot per inch resolution
    static SLstring  configPath;    //!< Default path for calibration files
    static SLstring  externalPath;  //!< Default path for external file storage
    static SLSceneID sceneID;       //!< ID of last loaded scene

    static map<string, string>         deviceParameter;    //! Generic device parameter
    static deque<function<void(void)>> jobsToBeThreaded;   //!< queue of functions to be executed in a thread
    static deque<function<void(void)>> jobsToFollowInMain; //!< queue of function to follow in the main thread
    static atomic<bool>                jobIsRunning;       //!< True if a parallel job is running

    static CVCalibrationEstimatorParams calibrationEstimatorParams;
    static CVCalibrationEstimator*      calibrationEstimator;
    static SLstring                     calibIniPath;  //!<thats where data/calibrations folder is located
    static SLstring                     calibFilePath; //!<thats where calibrations are stored and loaded from

    static const string CALIB_FTP_HOST; //!< ftp host for calibration up and download
    static const string CALIB_FTP_USER; //!< ftp login user for calibration up and download
    static const string CALIB_FTP_PWD;  //!< ftp login pwd for calibration up and download
    static const string CALIB_FTP_DIR;  //!< ftp directory for calibration up and download

private:
    static HighResTimer _timer;          //!< high precision timer
    static string       _jobProgressMsg; //!< Text message to show during progress
    static atomic<int>  _jobProgressNum; //!< Integer value to show progess
    static atomic<int>  _jobProgressMax; //!< Max. integer progress value
    static mutex        _jobMutex;       //!< Mutex to protect parallel access
};
//-----------------------------------------------------------------------------
#endif
