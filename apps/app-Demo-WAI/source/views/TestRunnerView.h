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

class TestRunnerView : protected SLSceneView
{
    struct TestData
    {
        std::string   mapFile;
        std::string   videoFile;
        CVCalibration calibration = {CVCameraType::VIDEOFILE, ""};
    };

    typedef std::string Location;
    typedef std::string Area;

    typedef std::vector<TestData>          TestDataVector;
    typedef std::map<Area, TestDataVector> AreaMap;
    typedef std::map<Location, AreaMap>    ErlebARTestSet;

    struct RelocalizationTestResult
    {
        bool  wasSuccessful;
        int   frameCount;
        int   relocalizationFrameCount;
        float ratio;
    };

    struct TrackingTestResult
    {
        bool  wasSuccessful;
        int   frameCount;
        int   trackingFrameCount;
        float ratio;
    };

    struct TestInstance
    {
        std::string location;
        std::string area;
        std::string video;
        std::string map;
        std::string calibration;
    };

public:
    enum TestMode
    {
        TestMode_None,
        TestMode_Relocalization,
        TestMode_Tracking
    };

    TestRunnerView(sm::EventHandler& eventHandler,
                   SLInputManager&   inputManager,
                   int               screenWidth,
                   int               screenHeight,
                   int               dotsPerInch,
                   std::string       erlebARDir,
                   std::string       calibDir,
                   std::string       fontPath,
                   std::string       configFile,
                   std::string       vocabularyFile,
                   std::string       imguiIniPath);

    bool start(TestMode      testMode,
               ExtractorType extractorType);
    bool update();

    // getters
    // TODO(dgj1): make these save to use
    std::string videoName() { return (_currentTestIndex < _testInstances.size()) ? _testInstances[_currentTestIndex].video : ""; }
    int         currentFrameIndex() { return _currentFrameIndex; }
    int         frameIndex() { return _frameCount; }

private:
    void launchTrackingTest(const Location& location,
                            const Area&     area,
                            TestDataVector& datas,
                            ExtractorType   extractorType,
                            std::string     vocabularyFile,
                            int             framerate = 0);

    TrackingTestResult runTrackingTest(std::string    videoFile,
                                       std::string    mapFile,
                                       std::string    vocFile,
                                       CVCalibration& calibration,
                                       ExtractorType  extractorType,
                                       int            framerate = 0);

    bool loadSites(const std::string&         erlebARDir,
                   const std::string&         configFile,
                   const std::string&         calibrationsDir,
                   std::vector<TestInstance>& testInstances);

    TestRunnerGui _gui;

    TestMode    _testMode;
    std::string _erlebARDir;
    std::string _calibDir;
    std::string _configFile;

    std::string   _vocFile;
    ExtractorType _extractorType;

    ErlebARTestSet            _erlebARTestSet;
    std::vector<TestInstance> _testInstances;

    // iterators
    int _currentTestIndex;

    // General test stuff
    int                          _frameCount;
    bool                         _testStarted = false;
    bool                         _testsDone   = false;
    SENSVideoStream*             _vStream     = nullptr;
    std::unique_ptr<KPextractor> _extractor;
    WAIMap*                      _map;
    int                          _currentFrameIndex;
    CVCalibration                _calibration = {CVCameraType::VIDEOFILE, ""};

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

    std::string testResults;

    // SL
    SLScene _scene;

    // slam
    FeatureExtractorFactory _featureExtractorFactory;
};

#endif