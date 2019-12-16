//#############################################################################
//  File:      WAISceneView.h
//  Purpose:   Node transform test application that demonstrates all transform
//             possibilities of SLNode
//  Author:    Marc Wacker
//  Date:      July 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef APP_WAI_SCENE_VIEW
#define APP_WAI_SCENE_VIEW

#include <vector>
#include "AppWAIScene.h"

#include <CVCalibration.h>
#include <WAIAutoCalibration.h>
#include <AppDirectories.h>
#include <WAIModeOrbSlam2.h>
#include <AppDemoWaiGui.h>

class AppDemoWaiGui;
class SLMaterial;
class SLPoints;
class SLNode;

struct OrbSlamStartResult
{
    bool        wasSuccessful;
    std::string errorString;
};

struct SlamParams
{
    std::string               videoFile;
    std::string               mapFile;
    std::string               calibrationFile;
    std::string               vocabularyFile;
    std::string               markerFile;
    WAI::ModeOrbSlam2::Params params;
};

//-----------------------------------------------------------------------------
class WAIApp
{
public:
    ~WAIApp();
    int load(int liveVideoTargetW, int liveVideoTargetH, int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi, AppDirectories* dirs);

    OrbSlamStartResult startOrbSlam(SlamParams* slamParams = nullptr);

    void onLoadWAISceneView(SLScene* s, SLSceneView* sv);
    bool update();
    bool updateTracking();

    void updateTrackingVisualization(const bool iKnowWhereIAm);

    void renderMapPoints(std::string                      name,
                         const std::vector<WAIMapPoint*>& pts,
                         SLNode*&                         node,
                         SLPoints*&                       mesh,
                         SLMaterial*&                     material);

    void renderKeyframes();
    void renderGraphs();

    void setupGUI(std::string appName, std::string configDir, int dotsPerInch);
    //void buildGUI(SLScene* s, SLSceneView* sv);
    //void openTest(std::string path);
    bool checkCalibration(const std::string& calibDir, const std::string& calibFileName);
    void setupDefaultErlebARDir();
    //static AppDemoGuiAbout* aboutDial;

    //AppDemoGuiError*          errorDial;
    static AppDirectories* dirs;

    static int                liveVideoTargetWidth;
    static int                liveVideoTargetHeight;
    static int                trackingImgWidth;
    static cv::Size2i         videoFrameSize;
    static float              videoFrameWdivH;
    static cv::VideoWriter*   videoWriter;
    static cv::VideoWriter*   videoWriterInfo;
    static WAI::ModeOrbSlam2* mode;
    static AppWAIScene*       waiScene;
    static bool               loaded;
    static SLGLTexture*       cpvrLogo;
    static SLGLTexture*       videoImage;
    static SLGLTexture*       testTexture;
    static ofstream           gpsDataStream;
    static SLQuat4f           lastKnowPoseQuaternion;
    static SLQuat4f           IMUQuaternion;

    static SlamParams* currentSlamParams;

    static bool resizeWindow;

    static std::string videoDir;
    static std::string calibDir;
    static std::string mapDir;
    static std::string vocDir;
    static std::string experimentsDir;

    static bool pauseVideo; // pause video file
    static int  videoCursorMoveIndex;

private:
    void                           close();
    std::unique_ptr<AppDemoWaiGui> _gui;
};

#endif
