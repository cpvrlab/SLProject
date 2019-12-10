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

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SL.h>
#include <SLApplication.h>
#include <SLScene.h>
#include <utility>

//-----------------------------------------------------------------------------
//! Global static objects
SLInputManager   SLApplication::inputManager;
SLScene*         SLApplication::scene = nullptr;
SLDeviceRotation SLApplication::devRot;
SLDeviceLocation SLApplication::devLoc;
SLstring         SLApplication::name    = "SLProjectApp";
SLstring         SLApplication::version = "2.4.000";
#ifdef _DEBUG
SLstring SLApplication::configuration = "Debug";
#else
SLstring SLApplication::configuration = "Release";
#endif
SLstring            SLApplication::computerUser  = "USER?";
SLstring            SLApplication::computerName  = "NAME?";
SLstring            SLApplication::computerBrand = "BRAND?";
SLstring            SLApplication::computerModel = "MODEL?";
SLstring            SLApplication::computerOS    = "OS?";
SLstring            SLApplication::computerOSVer = "OSVER?";
SLstring            SLApplication::computerArch  = "ARCH?";
SLstring            SLApplication::computerID    = "ID?";
SLstring            SLApplication::gitBranch     = SL_GIT_BRANCH;
SLstring            SLApplication::gitCommit     = SL_GIT_COMMIT;
SLstring            SLApplication::gitDate       = SL_GIT_DATE;
SLint               SLApplication::dpi           = 0;
map<string, string> SLApplication::deviceParameter;

CVCalibrationEstimatorParams SLApplication::calibrationEstimatorParams;
CVCalibrationEstimator*      SLApplication::calibrationEstimator = nullptr;
SLstring                     SLApplication::calibIniPath;
SLstring                     SLApplication::calibFilePath;

//! SLApplication::configPath is overwritten in slCreateAppAndScene.
SLstring                    SLApplication::configPath   = SLstring(SL_PROJECT_ROOT) + "/data/config/";
SLstring                    SLApplication::externalPath = SLstring(SL_PROJECT_ROOT) + "/data/config/";
SLSceneID                   SLApplication::sceneID      = SID_Empty;
deque<function<void(void)>> SLApplication::jobsToBeThreaded;
deque<function<void(void)>> SLApplication::jobsToFollowInMain;
atomic<bool>                SLApplication::jobIsRunning(false);
string                      SLApplication::_jobProgressMsg = "";
atomic<int>                 SLApplication::_jobProgressNum(0);
atomic<int>                 SLApplication::_jobProgressMax(0);
mutex                       SLApplication::_jobMutex;
HighResTimer                SLApplication::_timer;

const string SLApplication::CALIB_FTP_HOST = "pallas.bfh.ch:21";
const string SLApplication::CALIB_FTP_USER = "upload";
const string SLApplication::CALIB_FTP_PWD  = "FaAdbD3F2a";
const string SLApplication::CALIB_FTP_DIR  = "calibrations";

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

    name  = std::move(appName);
    scene = new SLScene(name, (cbOnSceneLoad)onSceneLoadCallback);
    _timer.start();
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

    delete scene;
    scene = nullptr;
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
SLstring SLApplication::getComputerInfos()
{
#if defined(SL_OS_WINDOWS) //..................................................

    // Computer user name
    const char* envvar = std::getenv("USER");
    computerUser       = envvar ? string(envvar) : "USER?";
    if (computerUser == "USER?")
    {
        const char* envvar = std::getenv("USERNAME");
        computerUser       = envvar ? string(envvar) : "USER?";
    }

    computerName = Utils::getHostName();

    // Get architecture
    SYSTEM_INFO siSysInfo;
    GetSystemInfo(&siSysInfo);
    switch (siSysInfo.wProcessorArchitecture)
    {
        case PROCESSOR_ARCHITECTURE_AMD64: computerArch = "x64"; break;
        case PROCESSOR_ARCHITECTURE_ARM: computerArch = "ARM"; break;
        case PROCESSOR_ARCHITECTURE_ARM64: computerArch = "ARM64"; break;
        case PROCESSOR_ARCHITECTURE_IA64: computerArch = "IA64"; break;
        case PROCESSOR_ARCHITECTURE_INTEL: computerArch = "x86"; break;
        default: computerArch = "???";
    }

    // Windows OS version
    OSVERSIONINFO osInfo;
    ZeroMemory(&osInfo, sizeof(OSVERSIONINFO));
    osInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
    GetVersionEx(&osInfo);
    char osVer[50];
    sprintf(osVer, "%u.%u", osInfo.dwMajorVersion, osInfo.dwMinorVersion);
    computerOSVer = string(osVer);

    computerBrand = "BRAND?";
    computerModel = "MODEL?";
    computerOS    = "Windows";

#elif defined(SL_OS_MACOS) //..................................................

    // Computer user name
    const char* envvar = std::getenv("USER");
    computerUser       = envvar ? string(envvar) : "USER?";
    if (computerUser == "USER?")
    {
        const char* envvar = std::getenv("USERNAME");
        computerUser       = envvar ? string(envvar) : "USER?";
    }

    computerName  = Utils::getHostName();
    computerBrand = "Apple";
    computerOS    = "MacOS";

    // Get MacOS version
    SInt32 majorV, minorV, bugfixV;
    Gestalt(gestaltSystemVersionMajor, &majorV);
    Gestalt(gestaltSystemVersionMinor, &minorV);
    Gestalt(gestaltSystemVersionBugFix, &bugfixV);
    char osVer[50];
    sprintf(osVer, "%d.%d.%d", majorV, minorV, bugfixV);
    computerOSVer = string(osVer);

    // Get model
    size_t len = 0;
    sysctlbyname("hw.model", nullptr, &len, nullptr, 0);
    char model[255];
    sysctlbyname("hw.model", model, &len, nullptr, 0);
    computerModel = model;

    computerArch = "ARCH?";

#elif defined(SL_OS_LINUX) //..................................................

    computerOS    = "Linux";
    computerUser  = "USER?";
    computerName  = Utils::getHostName();
    computerBrand = "BRAND?";
    computerModel = "MODEL?";
    computerOSVer = "OSVER?";
    computerArch  = "ARCH?";

#elif defined(SL_OS_MACIOS) //.................................................

    // Model and architecture are retrieved before in iOS under Objective C
    computerBrand      = "Apple";
    computerOS         = "iOS";
    const char* envvar = std::getenv("USER");
    computerUser       = envvar ? string(envvar) : "USER?";
    if (computerUser == "USER?")
    {
        const char* envvar = std::getenv("USERNAME");
        computerUser       = envvar ? string(envvar) : "USER?";
    }
    computerName = Utils::getHostName();

#elif defined(SL_OS_ANDROID) //................................................

    computerOS = "Android";

    /*
    "ro.build.version.release"     // * The user-visible version string. E.g., "1.0" or "3.4b5".
    "ro.build.version.incremental" // The internal value used by the underlying source control to represent this build.
    "ro.build.version.codename"    // The current development codename, or the string "REL" if this is a release build.
    "ro.build.version.sdk"         // The user-visible SDK version of the framework.

    "ro.product.model"             // * The end-user-visible name for the end product..
    "ro.product.manufacturer"      // The manufacturer of the product/hardware.
    "ro.product.board"             // The name of the underlying board, like "goldfish".
    "ro.product.brand"             // The brand (e.g., carrier) the software is customized for, if any.
    "ro.product.device"            // The name of the industrial design.
    "ro.product.name"              // The name of the overall product.
    "ro.hardware"                  // The name of the hardware (from the kernel command line or /proc).
    "ro.product.cpu.abi"           // The name of the instruction set (CPU type + ABI convention) of native code.
    "ro.product.cpu.abi2"          // The name of the second instruction set (CPU type + ABI convention) of native code.

    "ro.build.display.id"          // * A build ID string meant for displaying to the user.
    "ro.build.host"
    "ro.build.user"
    "ro.build.id"                  // Either a changelist number, or a label like "M4-rc20".
    "ro.build.type"                // The type of build, like "user" or "eng".
    "ro.build.tags"                // Comma-separated tags describing the build, like "unsigned,debug".
    */

    int len;

    char host[PROP_VALUE_MAX];
    len          = __system_property_get("ro.build.host", host);
    computerName = host ? string(host) : "NAME?";

    char user[PROP_VALUE_MAX];
    len          = __system_property_get("ro.build.user", user);
    computerUser = user ? string(user) : "USER?";

    char brand[PROP_VALUE_MAX];
    len           = __system_property_get("ro.product.brand", brand);
    computerBrand = string(brand);

    char model[PROP_VALUE_MAX];
    len           = __system_property_get("ro.product.model", model);
    computerModel = string(model);

    char osVer[PROP_VALUE_MAX];
    len           = __system_property_get("ro.build.version.release", osVer);
    computerOSVer = string(osVer);

    char arch[PROP_VALUE_MAX];
    len          = __system_property_get("ro.product.cpu.abi", arch);
    computerArch = string(arch);

#endif

    // build a unique as possible ID string that can be used in a filename
    computerID = computerUser + "-" + computerName + "-" + computerModel;
    computerID = Utils::replaceNonFilenameChars(computerID);
    std::replace(computerID.begin(), computerID.end(), '_', '-');
    return computerID;
}
//-----------------------------------------------------------------------------
