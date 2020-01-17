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

struct ExtractorIds
{
    int trackingExtractorId;
    int initializationExtractorId;
    int markerExtractorId;
};

struct SlamParams
{
    std::string               videoFile;
    std::string               mapFile;
    std::string               calibrationFile;
    std::string               vocabularyFile;
    std::string               markerFile;
    std::string               location;
    std::string               area;
    std::string               extractorID;
    WAI::ModeOrbSlam2::Params params;
    ExtractorIds              extractorIds;
};

enum WAIEventType
{
    WAIEventType_None,
    WAIEventType_StartOrbSlam,
    WAIEventType_SaveMap,
    WAIEventType_VideoControl,
    WAIEventType_VideoRecording,
    WAIEventType_MapNodeTransform,
    WAIEventType_SetExtractors
    //TODO(dgj1): rest of events
};

struct WAIEvent
{
    WAIEventType type;
};

struct WAIEventStartOrbSlam : WAIEvent
{
    WAIEventStartOrbSlam() { type = WAIEventType_StartOrbSlam; }

    SlamParams params;
};

struct WAIEventSaveMap : WAIEvent
{
    WAIEventSaveMap() { type = WAIEventType_SaveMap; }

    std::string location;
    std::string area;
    std::string marker;
};

struct WAIEventVideoControl : WAIEvent
{
    WAIEventVideoControl() { type = WAIEventType_VideoControl; }

    bool pauseVideo;
    int  videoCursorMoveIndex;
};

struct WAIEventVideoRecording : WAIEvent
{
    WAIEventVideoRecording() { type = WAIEventType_VideoRecording; }

    std::string filename;
};

struct WAIEventMapNodeTransform : WAIEvent
{
    WAIEventMapNodeTransform() { type = WAIEventType_MapNodeTransform; }

    SLTransformSpace tSpace;
    SLVec3f          rotation;
    SLVec3f          translation;
    float            scale;
};

struct WAIEventSetExtractors : WAIEvent
{
    WAIEventSetExtractors() { type = WAIEventType_SetExtractors; }

    ExtractorIds extractorIds;
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
    void        setDeviceParameter(const std::string& parameter,
                                   std::string        value);

    //sensor stuff (todo: move out of waiapp?)
    void setRotationQuat(float quatX,
                         float quatY,
                         float quatZ,
                         float quatW);
    void setLocationLLA(float latitudeDEG,
                        float longitudeDEG,
                        float altitudeM,
                        float accuracyM);
    bool usesRotationSensor();
    bool usesLocationSensor();

    //set path for external writable directory for mobile devices
    //todo: is this still needed?
    void initExternalDataDirectory(std::string path);

    WAI::ModeOrbSlam2* mode()
    {
        return _mode;
    }

    std::string videoDir;
    std::string calibDir;
    std::string mapDir;
    std::string vocDir;

    //video file editing
    bool doubleBufferedOutput;

    void updateVideoImage(cv::Mat frame);

private:
    bool updateTracking();
    bool initSLProject(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi);
    void loadWAISceneView(SLScene* s, SLSceneView* sv, std::string location, std::string area);

    void setupGUI(std::string appName, std::string configDir, int dotsPerInch);
    void setupDefaultErlebARDirTo(std::string dir);
    //!download all remote files to transferred directory
    void downloadCalibratinFilesTo(std::string dir);
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
    void saveMap(std::string location, std::string area, std::string marker);
    void transformMapNode(SLTransformSpace tSpace,
                          SLVec3f          rotation,
                          SLVec3f          translation,
                          float            scale);
    void handleEvents();

    cv::Ptr<cv::CLAHE> clahe;

    //todo: we dont need a pointer
    std::unique_ptr<AppWAIScene> _waiScene;
    WAI::ModeOrbSlam2*           _mode;
    SLSceneView*                 _sv         = nullptr;
    SLGLTexture*                 _videoImage = nullptr;

    SlamParams     _currentSlamParams;
    AppDirectories _dirs;

    //sensor stuff
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

    int   lastFrameIdx;
    CVMat undistortedLastFrame[2];

    // video controls
    bool _pauseVideo           = false;
    int  _videoCursorMoveIndex = 0;

    // video writer
    void saveVideo(std::string filename);
    void saveGPSData(std::string videofile);

    // slam params
    void                       setModeExtractors(ExtractorIds& extractorIds);
    KPextractor*               orbExtractor(int nf);
    KPextractor*               surfExtractor(int th);
    KPextractor*               glslExtractor(int nbKeypointsBigSigma, int nbKeypointsSmallSigma, float highThrs, float lowThrs, float bigSigma, float smallSigma);
    KPextractor*               kpExtractor(int id);
    std::map<int, std::string> extractorIdToNames;
    std::map<std::string, int> extractorNamesToIds;

    // event queue
    std::queue<WAIEvent*> eventQueue;
};

#endif
