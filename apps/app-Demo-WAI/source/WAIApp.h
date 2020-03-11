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
#include <DeviceData.h>
#include <WAIModeOrbSlam2.h>
#include <AppDemoWaiGui.h>
#include <SLInputEventInterface.h>
#include <WAISlam.h>
#include <SENSCamera.h>
#include <SENSVideoStream.h>
#include <GLSLextractor.h>
#include <FeatureExtractorFactory.h>
#include <SLInputManager.h>
#include <WAIEvent.h>
#include <SlamParams.h>

#include <states/SelectionState.h>
#include <states/StartUpState.h>

class SLMaterial;
class SLPoints;
class SLNode;
class AppDemoGuiError;
class AppDemoGuiSlamLoad;

class WAIApp : public SLInputEventInterface
{
public:
    enum class State
    {
        IDLE,
        SELECTION,
        START_UP,
        MAP_SCENE,
        TRACKING_SCENE,
        TRACKING_VIDEO_SCENE
    };

    WAIApp();
    ~WAIApp();
    //call load to correctly initialize wai app
    int  load(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi, AppDirectories directories);
    void setCamera(SENSCamera* camera);
    //try to load last slam configuration, else open loading dialog
    void loadSlam();

    //call update to update the frame, wai and visualization
    bool update();
    void close();
    void terminate();

    //initialize wai orb slam with transferred parameters
    void startOrbSlam(SlamParams slamParams);
    void showErrorMsg(std::string msg);

    //todo: replace when we are independent of SLApplication
    std::string            name();
    const SENSVideoStream* getVideoFileStream() const { return _videoFileStream.get(); }
    const CVCalibration&   getCalibration() const { return _calibration; }
    const cv::Size&        getFrameSize() const { return _videoFrameSize; }

    WAISlam* mode()
    {
        return _mode;
    }

    //std::string videoDir;
    //std::string mapDir;

private:
    void reset();
    void checkStateTransition();
    bool updateState();

    bool updateTracking(SENSFramePtr frame);
    int  initSLProject(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi);
    void loadWAISceneView(std::string location, std::string area);

    void setupGUI(std::string appName, std::string configDir, int dotsPerInch);
    void setupDefaultErlebARDirTo(std::string dir);
    //!download all remote files to transferred directory
    void downloadCalibrationFilesTo(std::string dir);

    void updateTrackingVisualization(const bool iKnowWhereIAm, cv::Mat& imgRGB);
    void saveMap(std::string location, std::string area, std::string marker);
    void transformMapNode(SLTransformSpace tSpace,
                          SLVec3f          rotation,
                          SLVec3f          translation,
                          float            scale);
    // video writer
    void saveVideo(std::string filename);
    //void saveGPSData(std::string videofile);

    void handleEvents();

    //get new frame from live video or video file stream
    //SENSFramePtr updateVideoOrCamera();
    SENSFramePtr getCameraFrame();
    SENSFramePtr getVideoFrame();

    //WAI::ModeOrbSlam2*           _mode;
    WAISlam*     _mode = nullptr;
    SLSceneView* _sv   = nullptr;

    SlamParams     _currentSlamParams;
    AppDirectories _dirs;
    //std::string    _calibDir;

    //sensor stuff
    //ofstream _gpsDataStream;
    //SLQuat4f _lastKnowPoseQuaternion;
    //SLQuat4f _IMUQuaternion;

    //load function has been called
    //bool _loaded  = false;
    //bool _started = false;

    cv::VideoWriter*                 _videoWriter = nullptr;
    std::unique_ptr<SENSVideoStream> _videoFileStream;
    SENSCamera*                      _camera = nullptr;

    cv::Size2i _videoFrameSize;

    std::unique_ptr<AppDemoWaiGui>      _gui;
    std::shared_ptr<AppDemoGuiError>    _errorDial;
    std::shared_ptr<AppDemoGuiSlamLoad> _guiSlamLoad;
    int                                 _lastFrameIdx;
    cv::Mat                             _undistortedLastFrame[2];
    bool                                _doubleBufferedOutput;

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

    AppWAIScene*   _waiScene = nullptr;
    std::string    _name;
    SLInputManager _inputManager;

    State           _state          = State::IDLE;
    SelectionState* _selectionState = nullptr;
    StartUpState    _startUpState;

    //load was called, we can switch to startup state
    bool _startUpRequested = false;
    //defines if ErlebAR scene was already selected or if user has to choose
    bool _selectErlebAR = true;
    //call start of erlebar state
    bool _startErlebAR = false;

    std::unique_ptr<DeviceData> _deviceData;
};

#endif
