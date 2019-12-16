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
#include <SLInputEventInterface.h>

class SLMaterial;
class SLPoints;
class SLNode;
class AppDemoGuiError;

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
class WAIApp : public SLInputEventInterface
{
public:
    WAIApp();
    ~WAIApp();
    //call load to correctly initialize wai app
    int load(int            liveVideoTargetW,
             int            liveVideoTargetH,
             int            scrWidth,
             int            scrHeight,
             float          scr2fbX,
             float          scr2fbY,
             int            dpi,
             AppDirectories dirs);
    //call update to update the frame, wai and visualization
    bool update();
    void close();

    //initialize wai orb slam with transferred parameters
    OrbSlamStartResult startOrbSlam(SlamParams* slamParams = nullptr);
    void               showErrorMsg(std::string msg);

    //todo: replace when we are independent of SLApplication
    std::string name();
    //bool        resizeWindow() { return _resizeWindow; }
    //void        windowResized() { _resizeWindow = false; }

    WAI::ModeOrbSlam2* mode() { return _mode; }

    std::string videoDir;
    std::string calibDir;
    std::string mapDir;
    std::string vocDir;
    std::string experimentsDir;
    //video file editing
    bool pauseVideo           = false;
    int  videoCursorMoveIndex = 0;

private:
    bool updateTracking();
    bool initSLProject(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi);
    void onLoadWAISceneView(SLScene* s, SLSceneView* sv);

    void setupGUI(std::string appName, std::string configDir, int dotsPerInch);
    void setupDefaultErlebARDir();
    bool checkCalibration(const std::string& calibDir, const std::string& calibFileName);
    bool updateSceneViews();

    void updateTrackingVisualization(const bool iKnowWhereIAm);
    void renderMapPoints(std::string                      name,
                         const std::vector<WAIMapPoint*>& pts,
                         SLNode*&                         node,
                         SLPoints*&                       mesh,
                         SLMaterial*&                     material);
    void renderKeyframes();
    void renderGraphs();

    //todo: we dont need a pointer
    std::unique_ptr<AppWAIScene> _waiScene;
    WAI::ModeOrbSlam2*           _mode;
    SLSceneView*                 _sv         = nullptr;
    SLGLTexture*                 _videoImage = nullptr;

    SlamParams     _currentSlamParams;
    AppDirectories _dirs;

    ofstream _gpsDataStream;
    SLQuat4f _lastKnowPoseQuaternion;
    SLQuat4f _IMUQuaternion;

    bool _loaded = false;

    // bool _resizeWindow;
    //todo: do we need a pointer
    cv::VideoWriter* _videoWriter     = nullptr;
    cv::VideoWriter* _videoWriterInfo = nullptr;

    int _liveVideoTargetWidth;
    int _liveVideoTargetHeight;

    cv::Size2i _videoFrameSize;
    float      _videoFrameWdivH;

    std::unique_ptr<AppDemoWaiGui> _gui;
    AppDemoGuiError*               _errorDial = nullptr;
};

#endif
