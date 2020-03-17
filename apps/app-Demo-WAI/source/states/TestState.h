#ifndef TEST_STATE_H
#define TEST_STATE_H

#include <states/State.h>
#include <AppWAIScene.h>
#include <SLSceneView.h>
#include <AppDemoWaiGui.h>
#include <SlamParams.h>
#include <AppDemoGuiSlamLoad.h>
#include <AppDemoGuiError.h>
#include <SENSVideoStream.h>
#include <CVCalibration.h>

class WAISlam;
class WAIEvent;
class SENSCamera;

class TestState : public State
{
public:
    TestState(SLInputManager& inputManager,
              SENSCamera*     camera,
              int             screenWidth,
              int             screenHeight,
              int             dotsPerInch,
              std::string     fontPath,
              std::string     configDir,
              std::string     vocabularyDir,
              std::string     calibDir,
              std::string     videoDir,
              ButtonPressedCB backButtonPressedCB);
    ~TestState();

    bool update() override;

protected:
    void doStart() override;

    //try to load last slam (without clicking for convenience)
    void tryLoadLastSlam();
    void setupGUI();
    void handleEvents();
    void loadWAISceneView(std::string location, std::string area);
    void saveMap(std::string location, std::string area, std::string marker);
    void saveVideo(std::string filename);
    void showErrorMsg(std::string msg);
    void startOrbSlam(SlamParams slamParams);
    void transformMapNode(SLTransformSpace tSpace,
                          SLVec3f          rotation,
                          SLVec3f          translation,
                          float            scale);
    void downloadCalibrationFilesTo(std::string dir);
    void updateVideoTracking();
    void updateTrackingVisualization(const bool iKnowWhereIAm, cv::Mat& imgRGB);
    void setupDefaultErlebARDirTo(std::string dir);

    //video
    CVCalibration                    _calibration = {CVCameraType::FRONTFACING, ""};
    SENSCamera*                      _camera      = nullptr;
    cv::VideoWriter*                 _videoWriter = nullptr;
    std::unique_ptr<SENSVideoStream> _videoFileStream;
    bool                             _pauseVideo           = false;
    int                              _videoCursorMoveIndex = 0;
    bool                             _showUndistorted      = true;
    cv::Size2i                       _videoFrameSize;

    int     _lastFrameIdx;
    cv::Mat _undistortedLastFrame[2];
    bool    _doubleBufferedOutput;

    //slam
    WAISlam*   _mode = nullptr;
    SlamParams _currentSlamParams;

    //scene
    AppWAIScene _s;
    SLSceneView _sv;

    //gui
    AppDemoWaiGui                       _gui;
    std::shared_ptr<AppDemoGuiSlamLoad> _guiSlamLoad;
    std::shared_ptr<AppDemoGuiError>    _errorDial;

    SLAssetManager _assets;

    std::string _configDir;
    std::string _vocabularyDir;
    std::string _calibDir;
    std::string _videoDir;

    std::queue<WAIEvent*> _eventQueue;

    FeatureExtractorFactory      _featureExtractorFactory;
    std::unique_ptr<KPextractor> _trackingExtractor;
    std::unique_ptr<KPextractor> _initializationExtractor;
    std::unique_ptr<KPextractor> _markerExtractor;
};

#endif //TEST_STATE_H
