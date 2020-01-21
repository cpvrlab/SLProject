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
#include <SENSCamera.h>
#include <SENSVideoStream.h>
#include <GLSLextractor.h>

class SLMaterial;
class SLPoints;
class SLNode;
class AppDemoGuiError;

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

class FeatureExtractorFactory
{

public:
    FeatureExtractorFactory()
    {
        _extractorIdToNames.push_back("SURF-BRIEF-500");
        _extractorIdToNames.push_back("SURF-BRIEF-800");
        _extractorIdToNames.push_back("SURF-BRIEF-1000");
        _extractorIdToNames.push_back("SURF-BRIEF-1200");
        _extractorIdToNames.push_back("FAST-ORBS-1000");
        _extractorIdToNames.push_back("FAST-ORBS-2000");
        _extractorIdToNames.push_back("FAST-ORBS-4000");
        _extractorIdToNames.push_back("GLSL-1");
        _extractorIdToNames.push_back("GLSL");
    }

    std::unique_ptr<KPextractor> make(int id, cv::Size videoFrameSize)
    {
        switch (id)
        {
            case 0:
                return std::move(surfExtractor(500));
            case 1:
                return std::move(surfExtractor(800));
            case 2:
                return std::move(surfExtractor(1000));
            case 3:
                return std::move(surfExtractor(1200));
            case 4:
                return std::move(orbExtractor(1000));
            case 5:
                return std::move(orbExtractor(2000));
            case 6:
                return std::move(orbExtractor(4000));
            case 7:
                return std::move(glslExtractor(videoFrameSize, 16, 16, 0.5, 0.25, 1.9, 1.4));
            case 8:
                return std::move(glslExtractor(videoFrameSize, 16, 16, 0.5, 0.25, 1.8, 1.2));
            default:
                return std::move(surfExtractor(1000));
        }
    }

    const std::vector<std::string>& getExtractorIdToNames() const
    {
        return _extractorIdToNames;
    }

private:
    std::unique_ptr<KPextractor> orbExtractor(int nf)
    {
        float fScaleFactor = 1.2;
        int   nLevels      = 8;
        int   fIniThFAST   = 20;
        int   fMinThFAST   = 7;
        return std::move(
          std::make_unique<ORB_SLAM2::ORBextractor>(nf, fScaleFactor, nLevels, fIniThFAST, fMinThFAST));
    }

    std::unique_ptr<KPextractor> surfExtractor(int th)
    {
        return std::move(
          std::make_unique<ORB_SLAM2::SURFextractor>(th));
    }

    std::unique_ptr<KPextractor> glslExtractor(const cv::Size& videoFrameSize, int nbKeypointsBigSigma, int nbKeypointsSmallSigma, float highThrs, float lowThrs, float bigSigma, float smallSigma)
    {
        // int nbKeypointsBigSigma, int nbKeypointsSmallSigma, float highThrs, float lowThrs, float bigSigma, float smallSigma
        return std::move(
          std::make_unique<GLSLextractor>(videoFrameSize.width, videoFrameSize.height, nbKeypointsBigSigma, nbKeypointsSmallSigma, highThrs, lowThrs, bigSigma, smallSigma));
    }

    std::vector<std::string> _extractorIdToNames;
};

//-----------------------------------------------------------------------------
class WAIApp : public SLInputEventInterface
{
public:
    WAIApp();
    ~WAIApp();
    //call load to correctly initialize wai app
    int load(SENSCamera*    camera,
             int            liveVideoTargetW,
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
    void startOrbSlam(SlamParams* slamParams = nullptr);
    void showErrorMsg(std::string msg);

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
    void                   initExternalDataDirectory(std::string path);
    const SENSVideoStream* getVideoFileStream() const { return _videoFileStream.get(); }
    const CVCalibration&   getCalibration() const { return _calibration; }
    const cv::Size&        getFrameSize() const { return _videoFrameSize; }

    WAI::ModeOrbSlam2* mode()
    {
        return _mode;
    }

    std::string videoDir;
    std::string calibDir;
    std::string mapDir;
    std::string vocDir;

private:
    //bool updateTracking();
    bool updateTracking(SENSFramePtr frame);
    bool initSLProject(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi);
    void loadWAISceneView(SLScene* s, SLSceneView* sv, std::string location, std::string area);

    void setupGUI(std::string appName, std::string configDir, int dotsPerInch);
    void setupDefaultErlebARDirTo(std::string dir);
    //!download all remote files to transferred directory
    void downloadCalibratinFilesTo(std::string dir);
    bool checkCalibration(const std::string& calibDir, const std::string& calibFileName);
    bool updateSceneViews();

    void updateTrackingVisualization(const bool iKnowWhereIAm, cv::Mat& imgRGB);
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
    // video writer
    void saveVideo(std::string filename);
    void saveGPSData(std::string videofile);

    void handleEvents();

    //get new frame from live video or video file stream
    SENSFramePtr updateVideoOrCamera();

    cv::Ptr<cv::CLAHE> _clahe;

    //todo: we dont need a pointer
    std::unique_ptr<AppWAIScene> _waiScene;
    WAI::ModeOrbSlam2*           _mode       = nullptr;
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
    cv::VideoWriter*                 _videoWriter     = nullptr;
    cv::VideoWriter*                 _videoWriterInfo = nullptr;
    std::unique_ptr<SENSVideoStream> _videoFileStream;
    SENSCamera*                      _camera = nullptr;

    int _liveVideoTargetWidth;
    int _liveVideoTargetHeight;

    cv::Size2i _videoFrameSize;
    float      _videoFrameWdivH;

    std::unique_ptr<AppDemoWaiGui> _gui;
    AppDemoGuiError*               _errorDial = nullptr;

    int     _lastFrameIdx;
    cv::Mat _undistortedLastFrame[2];
    bool    _doubleBufferedOutput;

    // video controls
    bool _pauseVideo           = false;
    int  _videoCursorMoveIndex = 0;

    // event queue
    std::queue<WAIEvent*> _eventQueue;

    CVCalibration _calibration     = {CVCameraType::FRONTFACING, ""};
    bool          _showUndistorted = true;

    FeatureExtractorFactory      _featureExtractorFactory;
    std::unique_ptr<KPextractor> _trackingExtractor;
    std::unique_ptr<KPextractor> _initializationExtractor;
    std::unique_ptr<KPextractor> _markerExtractor;
};

#endif
