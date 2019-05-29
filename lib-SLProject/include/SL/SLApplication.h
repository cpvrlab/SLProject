//#############################################################################
//  File:      SLApplication.h
//  Author:    Marcus Hudritsch
//  Date:      Februar 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLAPPLICATION_H
#define SLAPPLICATION_H

#include <SLDeviceLocation.h>
#include <SLDeviceRotation.h>
#include <SLInputManager.h>
#include <mutex>

using namespace std;

class SLScene;
class SLCVCalibration;

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
 (calibMainCam) and one for the selfie camera on mobile devices (calibScndCam).
 The pointer activeCalib points to the active one.
*/
class SLApplication
{
    public:
    static void createAppAndScene(SLstring appName,
                                  void*    onSceneLoadCallback);
    static void deleteAppAndScene();

    static SLScene*         scene;          //!< scene pointer
    static SLInputManager   inputManager;   //!< Input events manager
    static SLCVCalibration* activeCalib;    //!< Pointer to the active calibration
    static SLCVCalibration  calibMainCam;   //!< OpenCV calibration for main video camera
    static SLCVCalibration  calibScndCam;   //!< OpenCV calibration for secondary video camera
    static SLCVCalibration  calibVideoFile; //!< OpenCV calibration for simulation using a video file
    static SLDeviceRotation devRot;         //!< Mobile device rotation from IMU
    static SLDeviceLocation devLoc;         //!< Mobile device location from GPS

    static SLstring  name;         //!< Applcation name
    static SLstring  version;      //!< SLProject version string
    static SLstring  gitBranch;    //!< Current GIT branch
    static SLstring  gitCommit;    //!< Current GIT commit short hash id
    static SLstring  gitDate;      //!< Current GIT commit date
    static SLint     dpi;          //!< Current UI dot per inch resolution
    static SLstring  configPath;   //!< Default path for calibration files
    static SLstring  externalPath; //!< Default path for external file storage
    static SLSceneID sceneID;      //!< ID of last loaded scene

    static SLfloat dpmm() { return (float)dpi / 25.4f; } //!< return dots per mm

    // Parallel job handling (please read remarks on handleParallelJob)
    static void   handleParallelJob();

    static void   jobProgressMsg(string msg);
    static void   jobProgressNum(int num) { _jobProgressNum = num; }
    static void   jobProgressMax(int max) { _jobProgressMax = max; }
    static string jobProgressMsg();
    static int    jobProgressNum() { return _jobProgressNum; }
    static int    jobProgressMax() { return _jobProgressMax; }

    static deque<function<void(void)>> jobsToBeThreaded;   //!< queue of functions to be executed in a thread
    static deque<function<void(void)>> jobsToFollowInMain; //!< queue of function to follow in the main thread
    static atomic<bool>                jobIsRunning;       //!< True if a parallel job is running

    private:
    static string      _jobProgressMsg; //!< Text message to show during progress
    static atomic<int> _jobProgressNum; //!< Integer value to show progess
    static atomic<int> _jobProgressMax; //!< Max. integer progress value
    static mutex       _jobMutex;       //!< Mutex to protect parallel access
};
//-----------------------------------------------------------------------------
#endif
