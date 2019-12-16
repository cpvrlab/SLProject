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
    int load(int liveVideoTargetW, int liveVideoTargetH, int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi, AppDirectories dirs);

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

    bool checkCalibration(const std::string& calibDir, const std::string& calibFileName);
    void setupDefaultErlebARDir();
    bool resizeWindow() { return _resizeWindow; }
    void windowResized() { _resizeWindow = false; }
    //static AppDemoGuiAbout* aboutDial;

    //AppDemoGuiError*          errorDial;

    static WAI::ModeOrbSlam2* mode;

    static SLGLTexture* videoImage;
    static SLGLTexture* testTexture;

    static std::string videoDir;
    static std::string calibDir;
    static std::string mapDir;
    static std::string vocDir;
    static std::string experimentsDir;

    static bool pauseVideo; // pause video file
    static int  videoCursorMoveIndex;

private:
    AppDirectories _dirs;

    ofstream _gpsDataStream;
    SLQuat4f _lastKnowPoseQuaternion;
    SLQuat4f _IMUQuaternion;

    bool       _loaded = false;
    SlamParams _currentSlamParams;

    bool _resizeWindow;
    //todo: do we need a pointer
    cv::VideoWriter* _videoWriter     = nullptr;
    cv::VideoWriter* _videoWriterInfo = nullptr;

    //todo: we dont need a pointer
    std::unique_ptr<AppWAIScene> _waiScene;

    int _liveVideoTargetWidth;
    int _liveVideoTargetHeight;

    cv::Size2i _videoFrameSize;
    float      _videoFrameWdivH;

    void                           close();
    std::unique_ptr<AppDemoWaiGui> _gui;

    SLSceneView* _sv = nullptr;
};

#endif
