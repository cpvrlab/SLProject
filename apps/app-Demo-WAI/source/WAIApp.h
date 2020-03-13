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
#include <states/AreaTrackingState.h>
#include <states/CameraTestState.h>
#include <states/LocationMapState.h>
#include <states/TestState.h>
#include <states/TutorialState.h>

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
        /*!Wait for _startFromIdle to become true. When it is true start SelectionState for scene selection and when it is
        started switch to START_UP state or directly start StartUpScene if AppMode is already not AppMode::NONE.
        */
        IDLE,
        /*!In this state the user has to select a an AppMode. When selection is not NONE, we switch to state START_UP
        */
        SELECTION,
        /*!We start up the states depending on selected AppMode
        */
        START_UP,
        LOCATION_MAP,
        AREA_TRACKING,
        TEST,
        CAMERA_TEST,
        TUTORIAL
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
    //void startOrbSlam(SlamParams slamParams);
    //void showErrorMsg(std::string msg);

    //todo: replace when we are independent of SLApplication
    std::string name();
    //const SENSVideoStream* getVideoFileStream() const { return _videoFileStream.get(); }
    //const CVCalibration&   getCalibration() const { return _calibration; }
    //const cv::Size&        getFrameSize() const { return _videoFrameSize; }

    //WAISlam* mode()
    //{
    //    return _mode;
    //}

    //std::string videoDir;
    //std::string mapDir;

private:
    SENSCamera* getCamera();

    void reset();
    void checkStateTransition();
    bool updateState();

    //bool updateTracking(SENSFramePtr frame);
    //int  initSLProject(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi);
    //void loadWAISceneView(std::string location, std::string area);

    //void setupGUI(std::string appName, std::string configDir, int dotsPerInch);
    void setupDefaultErlebARDirTo(std::string dir);
    //!download all remote files to transferred directory
    //void downloadCalibrationFilesTo(std::string dir);

    //void updateTrackingVisualization(const bool iKnowWhereIAm, cv::Mat& imgRGB);
    //void saveMap(std::string location, std::string area, std::string marker);
    //void transformMapNode(SLTransformSpace tSpace,
    //SLVec3f          rotation,
    //SLVec3f          translation,
    //float            scale);
    // video writer
    /*void saveVideo(std::string filename);*/
    //void saveGPSData(std::string videofile);

    //void handleEvents();

    //get new frame from live video or video file stream
    //SENSFramePtr updateVideoOrCamera();
    SENSFramePtr getCameraFrame();
    //SENSFramePtr getVideoFrame();

    //WAI::ModeOrbSlam2*           _mode;
    //WAISlam*     _mode = nullptr;
    //SLSceneView* _sv   = nullptr;

    //SlamParams     _currentSlamParams;
    AppDirectories _dirs;
    //std::string    _calibDir;

    //sensor stuff
    //ofstream _gpsDataStream;
    //SLQuat4f _lastKnowPoseQuaternion;
    //SLQuat4f _IMUQuaternion;

    //load function has been called
    //bool _loaded  = false;
    //bool _started = false;

    //cv::VideoWriter*                 _videoWriter = nullptr;
    //std::unique_ptr<SENSVideoStream> _videoFileStream;
    SENSCamera* _camera = nullptr;
    //std::mutex  _cameraSetMutex;

    //cv::Size2i _videoFrameSize;

    //std::unique_ptr<AppDemoWaiGui>      _gui;
    //std::shared_ptr<AppDemoGuiError> _errorDial;
    //std::shared_ptr<AppDemoGuiSlamLoad> _guiSlamLoad;
    //int     _lastFrameIdx;
    //cv::Mat _undistortedLastFrame[2];
    //bool    _doubleBufferedOutput;

    //// video controls
    //bool _pauseVideo           = false;
    //int  _videoCursorMoveIndex = 0;

    // event queue
    //std::queue<WAIEvent*> _eventQueue;

    //CVCalibration _calibration     = {CVCameraType::FRONTFACING, ""};
    //bool          _showUndistorted = true;

    //FeatureExtractorFactory      _featureExtractorFactory;
    //std::unique_ptr<KPextractor> _trackingExtractor;
    //std::unique_ptr<KPextractor> _initializationExtractor;
    //std::unique_ptr<KPextractor> _markerExtractor;

    //AppWAIScene*   _waiScene = nullptr;
    std::string    _name;
    SLInputManager _inputManager;

    std::unique_ptr<DeviceData> _deviceData;

    State              _state             = State::IDLE;
    SelectionState*    _selectionState    = nullptr;
    StartUpState*      _startUpState      = nullptr;
    AreaTrackingState* _areaTrackingState = nullptr;
    CameraTestState*   _cameraTestState   = nullptr;
    LocationMapState*  _locationMapState  = nullptr;
    TestState*         _testState         = nullptr;
    TutorialState*     _tutorialState     = nullptr;

    //Sub-States that lead to state start() call:
    //set to true as soon as we have access to app resouces, then we can first visualize something
    bool _startFromIdle = false;
    //done after AppMode was selected, then we know what to start up
    bool _startFromStartUp = false;

    //defines if ErlebAR scene was already selected or if user has to choose
    AppMode _appMode = AppMode::NONE;
    //
    Area _area = Area::NONE;

    bool _switchToTracking = false;
};

#endif
