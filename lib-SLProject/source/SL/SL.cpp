//#############################################################################
//  File:      SL.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif
#include <SLCV.h>
#include <SLSceneView.h>
#include <SLCVCapture.h>
#include <SLDemoGui.h>

//-----------------------------------------------------------------------------
//! Default values for static fields
SLstring        SL::version             = "2.1.000";
SLstring        SL::configPath          = "../_data/config/";
SLstring        SL::configTime          = "-";
SLint           SL::dpi                 = 0;
SLfloat         SL::fontPropDots        = 0.0f;
SLfloat         SL::fontFixedDots       = 0.0f;
SLCommand       SL::currentSceneID      = C_sceneEmpty;
SLint           SL::testDurationSec     = 0;
SLint           SL::testFactor          = 1;
SLCommand       SL::testScene           = (SLCommand)-1;
SLCommand       SL::testSceneAll        = C_sceneMinimal;
SLLogVerbosity  SL::testLogVerbosity    = LV_quiet;
SLuint          SL::testFrameCounter    = 0;
SLbool          SL::showMenu            = true;
SLbool          SL::showAbout           = false;
SLbool          SL::showHelp            = false;
SLbool          SL::showHelpCalibration = false;
SLbool          SL::showCredits         = false;
SLbool          SL::showStatsTiming     = false;
SLbool          SL::showStatsScene      = false;
SLbool          SL::showStatsVideo      = false;
SLbool          SL::showInfosFrameworks = false;
SLbool          SL::showInfosScene      = false;
SLbool          SL::showSceneGraph      = false;
SLbool          SL::showProperties      = false;

SLstring SL::infoAbout =
"Welcome to the SLProject demo app. It is developed at the \
Computer Science Department of the Bern University of Applied Sciences. \
The app shows what you can learn in one semester about 3D computer graphics \
in real time rendering and ray tracing. The framework is developed \
in C++ with OpenGL ES so that it can run also on mobile devices. \
Ray tracing provides in addition high quality transparencies, reflections and soft shadows. \
Click to close and use the menu to choose different scenes and view settings. \
For more information please visit: https://github.com/cpvrlab/SLProject";

SLstring SL::infoCredits =
"Contributors since 2005 in alphabetic order: Martin Christen, Manuel Frischknecht, Michael \
Goettlicher, Timo Tschanz, Marc Wacker, Pascal Zingg \n\n\
Credits for external libraries:\n\
- assimp: assimp.sourceforge.net\n\
- imgui: github.com/ocornut/imgui\n\
- glew: glew.sourceforge.net\n\
- glfw: glfw.org\n\
- OpenCV: opencv.org\n\
- OpenGL: opengl.org";

SLstring SL::infoHelp =
"Help for mouse or finger control:\n\
- Use mouse or your finger to rotate the scene\n\
- Use mouse-wheel or pinch 2 fingers to go forward/backward\n\
- Use CTRL-mouse or 2 fingers to move sidewards/up-down\n\
- Double click or double tap to select object";

SLstring SL::infoCalibrate =
"The calibration process requires a chessboard image to be printed \
and glued on a flat board. You can find the PDF with the chessboard image on: \n\
https://github.com/cpvrlab/SLProject/tree/master/_data/calibrations/ \n\n\
For a calibration you have to take 20 images with detected inner \
chessboard corners. To take an image you have to click with the mouse \
or tap with finger into the screen. You can mirror the video image under \
Preferences > Video. \n\
After calibration the yellow wireframe cube should stick on the chessboard.";



//! Scene name string vector. Make sure they corrspond to the enum SLCommand
const SLVstring SL::testSceneNames = 
{
    "sceneAll                   ",
    "sceneMinimal               ",
    "sceneFigure                ",
    "sceneMeshLoad              ",
    "sceneVRSizeTest            ",
    "sceneLargeModel            ",
    "sceneRevolver              ",
    "sceneTextureFilter         ",
    "sceneTextureBlend          ",
    "sceneFrustumCull           ",
    "sceneMassiveData           ",
    "sceneShaderPerVertexBlinn  ",
    "sceneShaderPerPixelBlinn   ",
    "sceneShaderPerVertexWave   ",
    "sceneShaderWater           ",
    "sceneShaderBumpNormal      ",
    "sceneShaderBumpParallax    ",
    "sceneShaderEarth           ",
    "sceneTerrain               ",
    "sceneAnimationMass         ",
    "sceneAnimationSkeletal     ",
    "sceneAnimationNode         ",
    "sceneAnimationArmy         ",
    "sceneVideoTexture          ",
    "sceneVideoChristoffel      ",
    "sceneVideoCalibrateMain    ",
    "sceneVideoCalibrateScnd    ",
    "sceneVideoTrackChessMain   ",
    "sceneVideoTrackChessScnd   ",
    "sceneVideoTrackArucoMain   ",
    "sceneVideoTrackArucoScnd   ",
    "sceneVideoTrackFeat2DMain  ",
    "sceneVideoTrackFeat2DScnd  ",
    "sceneRTMuttenzerBox        ",
    "sceneRTSpheres             ",
    "sceneRTSoftShadows         ",
    "sceneRTDoF                 ",
    "sceneRTLens                ",
    "sceneRTTest                ",
    "sceneMaximal               "
};
//-----------------------------------------------------------------------------
//! SL::log
void SL::log(const char* format, ...)
{
    char log[4096];
    va_list argptr;
    va_start(argptr, format);
    vsprintf(log, format, argptr);
    va_end(argptr);

    #if defined(SL_OS_ANDROID)
    __android_log_print(ANDROID_LOG_INFO, "SLProject", log);
    #else
    cout << log;
    #endif
}
//-----------------------------------------------------------------------------
//! SL::Exit terminates the application with a message. No leak cheching.
void SL::exitMsg(const SLchar* msg, const SLint line, const SLchar* file)
{  
    #if defined(SL_OS_ANDROID)
    __android_log_print(ANDROID_LOG_INFO, "SLProject", 
                        "Exit %s at line %d in %s\n", msg, line, file);
    SL::log("Exit %s at line %d in %s\n", msg, line, file);
    #endif
   
    #ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
    // turn off leak checks on forced exit
    nvwa::new_autocheck_flag = false;
    #endif
   
    exit(-1);
}
//-----------------------------------------------------------------------------
//! SL::Warn message output
void SL::warnMsg(const SLchar* msg, const SLint line, const SLchar* file)
{  
    #if defined(SL_OS_ANDROID)
    __android_log_print(ANDROID_LOG_INFO, "SLProject", 
                        "Warning: %s at line %d in %s\n", msg, line, file);
    #else
    SL::log("Warning %s at line %d in %s\n", msg, line, file);
    #endif
}
//-----------------------------------------------------------------------------
/*! SL::maxThreads returns in release config the max. NO. of threads and in 
debug config 1. Try to avoid multithreading in the debug configuration. 
*/
SLuint SL::maxThreads()
{  
    #ifdef _DEBUG
    return 1;
    #else
    return SL_max(thread::hardware_concurrency(), 1U);
    #endif
}
//------------------------------------------------------------------------------
//! Parses the command line arguments and sets the according scene test variable.
/*! The following command line arguments can be passed:\n
- testScene:       scene int ID defined in the enum SLCommand (0=sceneAll)\n
- testDurationSec: test duration in int sec.\n
- testFactor:      test int factor used for scaling in SLScene::onLoad.\n
\n
Example:\
app_Demo.exe testScene=1 testDurationSec=5 testFactor=1\n
\n
Starts the app with scene 1 (=sceneMinimal) and runs for 5 sec. with testFactor 1.
If the testScene is 0 (=sceneAll) all scenes are tested one after the other.\n
The scenes are changed and logged in SLSceneView::testRunIsFinished() that is
called in SLSceneView::onPaint().

\param cmdLineArgs A string vector with all command line arguments.
*/
void SL::parseCmdLineArgs(SLVstring& cmdLineArgs)
{   // Default values
    SL::testScene = (SLCommand)-1;
    SL::testDurationSec = -1;

    SLVstring argComponents;
    for (SLstring arg : cmdLineArgs)
    {
        SLUtils::split(arg, '=', argComponents);
        if (argComponents.size()==2)
        {
            if(argComponents[0] ==  "testScene")
            {   SLint iScene = atoi(argComponents[1].c_str());
                if (iScene >= C_sceneAll && iScene <= C_sceneRTTest)
                {   SL::testScene = (SLCommand)iScene;
                    SL::testLogVerbosity = LV_normal;
                    if (SL::testScene == C_sceneAll)
                        SL::testSceneAll = C_sceneMinimal;
                    if (SL::testDurationSec == -1)
                        SL::testDurationSec = 5;
                }
            }

            if(argComponents[0] ==  "testDurationSec")
            {   SLint sec = atoi(argComponents[1].c_str());
                if (sec > 0) SL::testDurationSec = sec;
            }

            if(argComponents[0] ==  "testFactor")
            {   SLint factor = atoi(argComponents[1].c_str());
                if (factor > 0) SL::testFactor = factor;
            }
        }
        argComponents.clear();
    }
}
//-----------------------------------------------------------------------------
//! Loads the configuration from readable path
void SL::loadConfig(SLSceneView* sv)
{
    SLstring fullPathAndFilename = SL::configPath + "SLProject.yml";

    if (!SLFileSystem::fileExists(fullPathAndFilename))
        return;
        
    SLCVFileStorage fs(fullPathAndFilename, SLCVFileStorage::READ);
    
    if (!fs.isOpened())
    {   SL_LOG("Failed to open file for reading!");
        return;
    }

    SLint i; SLbool b;
    fs["configTime"]            >> SL::configTime;
    fs["dpi"]                   >> SL::dpi;
    fs["fontPropDots"]          >> SL::fontPropDots;
    fs["fontFixedDots"]         >> SL::fontFixedDots;
    fs["currentSceneID"]        >> i; SL::currentSceneID = (SLCommand)i;
    fs["showMenu"]              >> b; SL::showMenu = b;
    fs["showStatsTiming"]       >> b; SL::showStatsTiming = b;
    fs["showStatsMemory"]       >> b; SL::showStatsScene = b;
    fs["showStatsVideo"]        >> b; SL::showStatsVideo = b;
    fs["showInfosFrameworks"]   >> b; SL::showInfosFrameworks = b;
    fs["showInfosScene"]        >> b; SL::showInfosScene = b;
    fs["showSceneGraph"]        >> b; SL::showSceneGraph = b;
    fs["showProperties"]        >> b; SL::showProperties = b;
    fs["drawBits"]              >> i; sv->drawBits()->bits((SLuint)i);

    fs.release();
    SL_LOG("Config. loaded  : %s\n", fullPathAndFilename.c_str());
}
//-----------------------------------------------------------------------------
//! Saves the configuration to a writable path
void SL::saveConfig(SLSceneView* sv)
{ 
    SLstring fullPathAndFilename = SL::configPath + "SLProject.yml";
    SLCVFileStorage fs(fullPathAndFilename, SLCVFileStorage::WRITE);
    
    if (!fs.isOpened())
    {   SL_EXIT_MSG("Failed to open file for writing!");
        return;
    }
     
    fs << "configTime"              << SLUtils::getLocalTimeString();
    fs << "dpi"                     << SL::dpi;
    fs << "fontPropDots"            << SL::fontPropDots;
    fs << "fontFixedDots"           << SL::fontFixedDots;
    fs << "currentSceneID"          << (SLint)SL::currentSceneID;
    fs << "showMenu"                << SL::showMenu;
    fs << "showStatsTiming"         << SL::showStatsTiming;
    fs << "showStatsMemory"         << SL::showStatsScene;
    fs << "showStatsVideo"          << SL::showStatsVideo;
    fs << "showInfosFrameworks"     << SL::showInfosFrameworks;
    fs << "showInfosScene"          << SL::showInfosScene;
    fs << "showSceneGraph"          << SL::showSceneGraph;
    fs << "showProperties"          << SL::showProperties;
    fs << "drawBits"                << (SLint)sv->drawBits()->bits();

    fs.release();
    SL_LOG("Config. saved   : %s\n", fullPathAndFilename.c_str());
}
//-----------------------------------------------------------------------------
