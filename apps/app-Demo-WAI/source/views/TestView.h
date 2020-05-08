#ifndef TEST_VIEW_H
#define TEST_VIEW_H

#include <AppWAIScene.h>
#include <SLSceneView.h>
#include <SLTransformNode.h>
#include <AppDemoWaiGui.h>
#include <SlamParams.h>
#include <AppDemoGuiSlamLoad.h>
#include <SENSVideoStream.h>
#include <CVCalibration.h>
#include <queue>
#include <ImageBuffer.h>

class WAISlam;
struct WAIEvent;
class SENSCamera;

class TestView : protected SLSceneView
{
public:
    TestView(sm::EventHandler& eventHandler,
             SLInputManager&   inputManager,
             SENSCamera*       camera,
             int               screenWidth,
             int               screenHeight,
             int               dotsPerInch,
             std::string       fontPath,
             std::string       configDir,
             std::string       vocabularyDir,
             std::string       calibDir,
             std::string       videoDir);
    ~TestView();

    bool update();
    //try to load slam params and start slam
    void start();
    void postStart();
    //call when view becomes visible
    void show() { _gui.onShow(); }

protected:
    void tryLoadLastSlam();
    void handleEvents();
    void loadWAISceneView(std::string location, std::string area);
    void saveMap(std::string location, std::string area, std::string marker);
    void saveVideo(std::string filename);
    void startOrbSlam(SlamParams slamParams);
    void transformMapNode(SLTransformSpace tSpace,
                          SLVec3f          rotation,
                          SLVec3f          translation,
                          float            scale);
    void downloadCalibrationFilesTo(std::string dir);

    void updateVideoTracking();
    void updateTrackingVisualization(const bool iKnowWhereIAm, cv::Mat& imgRGB);
    void setupDefaultErlebARDirTo(std::string dir);
    void startAsync();

    //video
    CVCalibration                    _calibration = {CVCameraType::FRONTFACING, ""};
    SENSCamera*                      _camera      = nullptr;
    cv::VideoWriter*                 _videoWriter = nullptr;
    std::unique_ptr<SENSVideoStream> _videoFileStream;
    bool                             _pauseVideo           = false;
    int                              _videoCursorMoveIndex = 0;
    bool                             _showUndistorted      = true;
    cv::Size2i                       _videoFrameSize;

    std::vector<std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>>> _calibrationMatchings;

    //slam
    WAISlam*   _mode = nullptr;
    SlamParams _currentSlamParams;

    FeatureExtractorFactory      _featureExtractorFactory;
    std::unique_ptr<KPextractor> _trackingExtractor;
    std::unique_ptr<KPextractor> _initializationExtractor;
    std::unique_ptr<KPextractor> _markerExtractor;
    ImageBuffer                  _imgBuffer;

    std::queue<WAIEvent*> _eventQueue;

    //scene
    AppWAIScene _scene;

    SLAssetManager _assets;

    std::string _configDir;
    std::string _vocabularyDir;
    std::string _calibDir;
    std::string _videoDir;

    std::thread _startThread;
    std::thread _calibrationThread;
    bool        _isCalibrated;

    SLTransformNode* _transformationNode = nullptr;

    //gui (declaration down here because it depends on a lot of members in initializer list of constructor)
    AppDemoWaiGui _gui;
};

#endif //TEST_VIEW_H
