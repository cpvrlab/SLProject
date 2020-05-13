#ifndef TEST_RUNNER_VIEW_H
#define TEST_RUNNER_VIEW_H

#include <SlamParams.h>
#include <SLSceneView.h>
#include <CVCalibration.h>
#include <TestRunnerGui.h>
#include <sm/EventHandler.h>
#include <ErlebAR.h>
#include <KPextractor.h>
#include <SENSVideoStream.h>
#include <fbow.h>

class TestRunnerView : protected SLSceneView
{
    struct TestInstance
    {
        std::string location;
        std::string area;
        std::string video;
        std::string map;
        std::string calibration;
        std::string extractorType;
    };

public:
    enum TestMode
    {
        TestMode_None,
        TestMode_Relocalization,
        TestMode_Tracking
    };

    TestRunnerView(sm::EventHandler&   eventHandler,
                   SLInputManager&     inputManager,
                   const ImGuiEngine&  imGuiEngine,
                   ErlebAR::Resources& resources,
                   int                 screenWidth,
                   int                 screenHeight,
                   int                 dotsPerInch,
                   std::string         erlebARDir,
                   std::string         calibDir,
                   std::string         fontPath,
                   std::string         configFile,
                   std::string         vocabularyFile,
                   std::string         imguiIniPath);

    bool start(TestMode testMode);
    bool update();

    // getters
    int         testIndex() { return (_currentTestIndex + 1); }
    int         testCount() { return _testInstances.size(); }
    std::string location() { return (_currentTestIndex < _testInstances.size()) ? _testInstances[_currentTestIndex].location : ""; }
    std::string area() { return (_currentTestIndex < _testInstances.size()) ? _testInstances[_currentTestIndex].area : ""; }
    std::string video() { return (_currentTestIndex < _testInstances.size()) ? _testInstances[_currentTestIndex].video : ""; }
    int         currentFrameIndex() { return _currentFrameIndex; }
    int         frameIndex() { return _frameCount; }
    bool        testsRunning() { return _testStarted; }

private:
    bool loadSites(const std::string&         erlebARDir,
                   const std::string&         configFile,
                   const std::string&         calibrationsDir,
                   std::vector<TestInstance>& testInstances);

    TestRunnerGui _gui;

    std::string _ftpHost;
    std::string _ftpUser;
    std::string _ftpPwd;
    std::string _ftpDir;

    TestMode    _testMode;
    std::string _erlebARDir;
    std::string _calibDir;
    std::string _configFile;

    std::vector<TestInstance> _testInstances;

    // iterators
    int _currentTestIndex;

    // General test stuff
    int                          _frameCount;
    bool                         _testStarted = false;
    SENSVideoStream*             _vStream     = nullptr;
    std::unique_ptr<KPextractor> _extractor;
    WAIMap*                      _map;
    int                          _currentFrameIndex;
    CVCalibration                _calibration = {CVCameraType::VIDEOFILE, ""};
    fbow::Vocabulary             _voc;
    bool                         _videoWasDownloaded;
    float                        _summedTime;

    // Relocalization test stuff
    int _relocalizationFrameCount;

    // Tracking test stuff
    WAIFrame      _lastFrame;
    bool          _isTracking;
    bool          _relocalizeOnce;
    cv::Mat       _extrinsic;
    LocalMap      _localMap;
    int           _trackingFrameCount;
    int           _maxTrackingFrameCount;
    LocalMapping* _localMapping;
    LoopClosing*  _loopClosing;
    int           _inliers;
    cv::Mat       _velocity;
    unsigned long _lastKeyFrameFrameId;
    unsigned int  _lastRelocFrameId;

    std::string _testResults;

    // SL
    SLScene _scene;

    // slam
    FeatureExtractorFactory _featureExtractorFactory;
};

#endif