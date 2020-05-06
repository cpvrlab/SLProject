#include <views/TestRunnerView.h>
#include <SENSCamera.h>
#include <WAIMapStorage.h>
#include <AppWAISlamParamHelper.h>
#include <Utils.h>
#include <FtpUtils.h>

TestRunnerView::TestRunnerView(sm::EventHandler& eventHandler,
                               SLInputManager&   inputManager,
                               int               screenWidth,
                               int               screenHeight,
                               int               dotsPerInch,
                               std::string       erlebARDir,
                               std::string       calibDir,
                               std::string       fontPath,
                               std::string       configFile,
                               std::string       vocabularyFile,
                               std::string       imguiIniPath)
  : SLSceneView(&_scene, dotsPerInch, inputManager),
    _gui(eventHandler, dotsPerInch, fontPath),
    _scene("TestRunnerScene", nullptr),
    _testMode(TestMode_None),
    _erlebARDir(erlebARDir),
    _calibDir(calibDir),
    _configFile(configFile),
    _vocFile(vocabularyFile),
    _localMapping(nullptr),
    _loopClosing(nullptr)
{
    init("TestRunnerView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    onInitialize();
}

bool TestRunnerView::start(TestMode      testMode,
                           ExtractorType extractorType)
{
    bool result = false;

    if (loadSites(_erlebARDir, _configFile, _calibDir, _testInstances))
    {
        _testMode         = testMode;
        _currentTestIndex = 0;

        _testsDone   = false;
        _testStarted = true;

        _extractorType = extractorType;

        result = true;
    }

    return result;
}

bool TestRunnerView::update()
{
    if (_testStarted)
    {
        SENSFramePtr sensFrame;
        if (_vStream && (sensFrame = _vStream->grabNextFrame()))
        {
            cv::Mat intrinsic  = _calibration.cameraMat();
            cv::Mat distortion = _calibration.distortion();

            WAIFrame currentFrame = WAIFrame(sensFrame.get()->imgGray,
                                             0.0f,
                                             _extractor.get(),
                                             intrinsic,
                                             distortion,
                                             WAIOrbVocabulary::get(),
                                             false);

            switch (_testMode)
            {
                case TestMode_Relocalization: {
                    int      inliers;
                    LocalMap localMap;
                    localMap.keyFrames.clear();
                    localMap.mapPoints.clear();
                    localMap.refKF = nullptr;
                    if (WAISlam::relocalization(currentFrame, _map, localMap, inliers))
                    {
                        _relocalizationFrameCount++;
                    }
                }
                break;

                case TestMode_Tracking: {
                    if (_isTracking)
                    {
                        if (WAISlamTools::tracking(_map, _localMap, currentFrame, _lastFrame, _lastRelocFrameId, _velocity, _inliers))
                        {
                            _trackingFrameCount++;
                            WAISlamTools::motionModel(currentFrame, _lastFrame, _velocity, _extrinsic);
                            WAISlamTools::serialMapping(_map, _localMap, _localMapping, _loopClosing, currentFrame, _inliers, _lastRelocFrameId, _lastKeyFrameFrameId);
                        }
                        else
                        {
                            if (_trackingFrameCount > _maxTrackingFrameCount)
                            {
                                _maxTrackingFrameCount = _trackingFrameCount;
                            }
                            _trackingFrameCount = 0;
                            _isTracking == false;
                        }
                    }
                    else
                    {
                        int inliers;
                        if (WAISlam::relocalization(currentFrame, _map, _localMap, inliers))
                        {
                            _isTracking     = true;
                            _relocalizeOnce = true;

                            WAISlamTools::motionModel(currentFrame, _lastFrame, _velocity, _extrinsic);
                            WAISlamTools::serialMapping(_map, _localMap, _localMapping, _loopClosing, currentFrame, inliers, _lastRelocFrameId, _lastKeyFrameFrameId);
                        }
                    }

                    _lastFrame = WAIFrame(currentFrame);
                }
                break;
            }

            _currentFrameIndex++;
        }
        else
        {
            if (_vStream)
            {
                TestRunnerView::TestInstance currentTest = _testInstances[_currentTestIndex];

                switch (_testMode)
                {
                    case TestMode_Relocalization: {
                        testResults +=
                          currentTest.location + ";" +
                          currentTest.area + ";" +
                          currentTest.video + ";" +
                          currentTest.map + ";" +
                          std::to_string(_currentFrameIndex) + ";" +
                          std::to_string(_relocalizationFrameCount) + ";" +
                          Utils::toString((float)_relocalizationFrameCount / (float)_currentFrameIndex, 2) + "\n";
                    }
                    break;

                    case TestMode_Tracking: {
                        testResults +=
                          currentTest.location + ";" +
                          currentTest.area + ";" +
                          currentTest.video + ";" +
                          currentTest.map + ";" +
                          std::to_string(_currentFrameIndex) + ";" +
                          std::to_string(_maxTrackingFrameCount) + ";" +
                          Utils::toString((float)_maxTrackingFrameCount / (float)_currentFrameIndex, 2) + "\n";
                    }
                    break;
                }

                _currentTestIndex++;
            }

            if (_currentTestIndex == _testInstances.size())
            {
                // done with current tests
                _testStarted = false;
                _testsDone   = true;

                std::string resultDir = _erlebARDir + "TestRunner/";
                if (!Utils::dirExists(resultDir))
                    Utils::makeDirRecurse(resultDir);

                std::string testModeString = "none";
                switch (_testMode)
                {
                    case TestMode_Relocalization: {
                        testModeString = "relocalization";
                    }
                    break;

                    case TestMode_Tracking: {
                        testModeString = "tracking";
                    }
                    break;
                }

                // save results in file
                std::string resultFileName = Utils::getDateTime2String() + "_" + testModeString + "_results.csv";
                std::string resultFile     = resultDir + resultFileName;

                std::ofstream f;
                f.open(resultFile, std::ofstream::out);

                if (!f.is_open())
                {
                    Utils::log("WAI", "TestRunner::update: Could not open result file %s", resultFile.c_str());
                    return false;
                }

                f << testResults;

                f.flush();
                f.close();

                // upload results to pallas
                const std::string ftpHost = "pallas.bfh.ch:21";
                const std::string ftpUser = "upload";
                const std::string ftpPwd  = "FaAdbD3F2a";
                const std::string ftpDir  = "erleb-AR/TestRunner/";

                std::string errorMsg;
                if (!FtpUtils::uploadFile(resultDir,
                                          resultFileName,
                                          ftpHost,
                                          ftpUser,
                                          ftpPwd,
                                          ftpDir,
                                          errorMsg))
                {
                    Utils::log("WAI", "TestRunner::update: Could not upload results file to pallas %s", errorMsg.c_str());
                }
            }
            else
            {
                const std::string ftpHost = "pallas.bfh.ch:21";
                const std::string ftpUser = "upload";
                const std::string ftpPwd  = "FaAdbD3F2a";
                const std::string ftpDir  = "erleb-AR/";

                bool instanceFound = false;
                while (_currentTestIndex < _testInstances.size() && !instanceFound)
                {
                    // download files for next test instance
                    TestRunnerView::TestInstance testInstance    = _testInstances[_currentTestIndex];
                    std::string                  testInstanceDir = "locations/" +
                                                  testInstance.location + "/" +
                                                  testInstance.area + "/";

                    std::string mapDir  = testInstanceDir + "maps/";
                    std::string mapFile = _erlebARDir + mapDir + testInstance.map;

                    if (!fileExists(mapFile))
                    {
                        std::string errorMsg;
                        Utils::makeDirRecurse(_erlebARDir + mapDir);
                        if (!FtpUtils::downloadFile(_erlebARDir + mapDir,
                                                    testInstance.map,
                                                    ftpHost,
                                                    ftpUser,
                                                    ftpPwd,
                                                    ftpDir + mapDir,
                                                    errorMsg))
                        {
                            Utils::log("WAI", "TestRunner::loadSites: Failed to load map file %s: %s", testInstance.map.c_str(), errorMsg.c_str());
                            _currentTestIndex++;
                            continue;
                        }
                    }

                    std::string videoDir  = testInstanceDir + "videos/";
                    std::string videoFile = _erlebARDir + videoDir + testInstance.video;
                    if (!fileExists(videoFile))
                    {
                        std::string errorMsg;
                        Utils::makeDirRecurse(_erlebARDir + videoDir);
                        if (!FtpUtils::downloadFile(_erlebARDir + videoDir,
                                                    testInstance.video,
                                                    ftpHost,
                                                    ftpUser,
                                                    ftpPwd,
                                                    ftpDir + videoDir,
                                                    errorMsg))
                        {
                            Utils::log("WAI", "TestRunner::loadSites: Failed to load video file %s: %s", testInstance.video.c_str(), errorMsg.c_str());
                            _currentTestIndex++;
                            continue;
                        }
                    }

                    std::string calibrationFile = _calibDir + testInstance.calibration;
                    if (!Utils::fileExists(calibrationFile))
                    {
                        std::string errorMsg;
                        if (!FtpUtils::downloadFile(_calibDir,
                                                    testInstance.calibration,
                                                    ftpHost,
                                                    ftpUser,
                                                    ftpPwd,
                                                    ftpDir + "calibrations/",
                                                    errorMsg))
                        {
                            Utils::log("WAI", "TestRunner::loadSites: Calibration file does not exist %s: %s", calibrationFile.c_str(), errorMsg.c_str());
                            _currentTestIndex++;
                            continue;
                        }
                    }

                    //load calibration file and check for aspect ratio
                    if (!_calibration.load(_calibDir, testInstance.calibration, true))
                    {
                        Utils::log("WAI", "TestRunner::loadSites: Could not load calibration file: %s", calibrationFile.c_str());
                        _currentTestIndex++;
                        continue;
                    }

                    SlamVideoInfos slamVideoInfos;
                    if (!extractSlamVideoInfosFromFileName(testInstance.video, &slamVideoInfos))
                    {
                        Utils::log("WAI", "TestRunner::loadSites: Could not extract slam video infos: %s", testInstance.video.c_str());
                        _currentTestIndex++;
                        continue;
                    }

                    std::vector<std::string> size;
                    Utils::splitString(slamVideoInfos.resolution, 'x', size);
                    if (size.size() == 2)
                    {
                        int width  = std::stoi(size[0]);
                        int height = std::stoi(size[1]);
                        if (_calibration.imageSize().width != width ||
                            _calibration.imageSize().height != height)
                        {
                            _calibration.adaptForNewResolution(CVSize(width, height), true);
                        }
                    }
                    else
                    {
                        Utils::log("WAI", "TestRunner::loadSites: Could not adapt calibration resolution");
                        _currentTestIndex++;
                        continue;
                    }

                    WAIFrame::mbInitialComputations = true;

                    WAIOrbVocabulary::initialize(_vocFile);
                    ORBVocabulary* orbVoc     = WAIOrbVocabulary::get();
                    WAIKeyFrameDB* keyFrameDB = new WAIKeyFrameDB(*orbVoc);

                    _map = new WAIMap(keyFrameDB);
                    WAIMapStorage::loadMap(_map, nullptr, orbVoc, mapFile, false, true);

                    if (_localMapping) delete _localMapping;
                    if (_loopClosing) delete _loopClosing;

                    if (_testMode == TestMode_Tracking)
                    {
                        _lastKeyFrameFrameId   = 0;
                        _lastRelocFrameId      = 0;
                        _inliers               = 0;
                        _trackingFrameCount    = 0;
                        _maxTrackingFrameCount = 0;

                        _isTracking     = false;
                        _relocalizeOnce = false;

                        _localMap.keyFrames.clear();
                        _localMap.mapPoints.clear();
                        _localMap.refKF = nullptr;

                        _localMapping = new ORB_SLAM2::LocalMapping(_map, 1, orbVoc, 0.95);
                        _loopClosing  = new ORB_SLAM2::LoopClosing(_map, orbVoc, false, false);

                        _localMapping->SetLoopCloser(_loopClosing);
                        _loopClosing->SetLocalMapper(_localMapping);
                    }

                    if (_vStream)
                        delete _vStream;

                    _vStream = new SENSVideoStream(videoFile, false, false, false);

                    CVSize2i videoSize       = _vStream->getFrameSize();
                    float    widthOverHeight = (float)videoSize.width / (float)videoSize.height;
                    _extractor               = _featureExtractorFactory.make(_extractorType, {videoSize.width, videoSize.height});

                    _frameCount               = _vStream->frameCount();
                    _currentFrameIndex        = 0;
                    _relocalizationFrameCount = 0;
                    _lastFrame                = WAIFrame();

                    instanceFound = true;
                }
            }
        }
    }

    bool result = onPaint();
    return result;
}

void TestRunnerView::launchTrackingTest(const Location&        location,
                                        const Area&            area,
                                        std::vector<TestData>& datas,
                                        ExtractorType          extractorType,
                                        std::string            vocabularyFile,
                                        int                    framerate)
{
    Utils::log("info", "TestRunnerView::lauchTest: Starting tracking test for area: %s", area.c_str());
    //the lastly saved map file (only valid if initialized is true)
    bool        initialized = false;
    std::string currentMapFileName;

    const float cullRedundantPerc = 0.99f;

    if (datas.size())
    {
        const float cullRedundantPerc = 0.95f;
        //select one calibration (we need one to instantiate mode and we need mode to load map)
        for (TestData testData : datas)
        {
            TrackingTestResult r = runTrackingTest(testData.videoFile, testData.mapFile, vocabularyFile, testData.calibration, extractorType, framerate);

            if (r.wasSuccessful)
            {
                Utils::log("info",
                           "%s;%s;%s;%i;%i;%.2f\n",
                           location.c_str(),
                           testData.videoFile.c_str(),
                           testData.mapFile.c_str(),
                           r.frameCount,
                           r.trackingFrameCount,
                           r.ratio);
            }
            else
            {
                Utils::log("warn", "TestRunnerView::launchTrackingTest: Never able to start traking");
            }
        }
    }
    else
    {
        Utils::log("warn", "TestRunnerView::launchTrackingTest: No tracking test for area: %s", area.c_str());
    }

    Utils::log("info", "TestRunnerView::launchTrackingTest: Finished tracking test for area: %s", area.c_str());
}

TestRunnerView::TrackingTestResult TestRunnerView::runTrackingTest(std::string    videoFile,
                                                                   std::string    mapFile,
                                                                   std::string    vocFile,
                                                                   CVCalibration& calibration,
                                                                   ExtractorType  extractorType,
                                                                   int            framerate)
{
    TrackingTestResult result = {};

    WAIFrame::mbInitialComputations = true;

    WAIOrbVocabulary::initialize(vocFile);
    ORBVocabulary* voc        = WAIOrbVocabulary::get();
    WAIKeyFrameDB* keyFrameDB = new WAIKeyFrameDB(*voc);

    WAIMap* map = new WAIMap(keyFrameDB);
    WAIMapStorage::loadMap(map, nullptr, voc, mapFile, false, true);

    LocalMapping* localMapping = new ORB_SLAM2::LocalMapping(map, 1, voc, 0.95);
    LoopClosing*  loopClosing  = new ORB_SLAM2::LoopClosing(map, voc, false, false);

    localMapping->SetLoopCloser(loopClosing);
    loopClosing->SetLocalMapper(localMapping);

    SENSVideoStream              vstream(videoFile, false, false, false, framerate);
    CVSize2i                     videoSize       = vstream.getFrameSize();
    float                        widthOverHeight = (float)videoSize.width / (float)videoSize.height;
    std::unique_ptr<KPextractor> extractor       = _featureExtractorFactory.make(extractorType, {videoSize.width, videoSize.height});

    cv::Mat       extrinsic;
    cv::Mat       intrinsic  = calibration.cameraMat();
    cv::Mat       distortion = calibration.distortion();
    cv::Mat       velocity;
    unsigned long lastKeyFrameFrameId = 0;
    unsigned int  lastRelocFrameId    = 0;
    int           inliers             = 0;
    int           frameCount          = 0;
    int           trackingFrameCount  = 0;

    int maxTrackingFrameCount = 0;

    bool     isTracking     = false;
    bool     relocalizeOnce = false;
    LocalMap localMap;
    localMap.keyFrames.clear();
    localMap.mapPoints.clear();
    localMap.refKF = nullptr;

    WAIFrame lastFrame = WAIFrame();

    while (SENSFramePtr sensFrame = vstream.grabNextResampledFrame())
    {
        WAIFrame frame = WAIFrame(sensFrame.get()->imgGray,
                                  0.0f,
                                  extractor.get(),
                                  intrinsic,
                                  distortion,
                                  voc,
                                  false);
        if (isTracking)
        {
            if (WAISlamTools::tracking(map, localMap, frame, lastFrame, lastRelocFrameId, velocity, inliers))
            {
                trackingFrameCount++;
                WAISlamTools::motionModel(frame, lastFrame, velocity, extrinsic);
                WAISlamTools::serialMapping(map, localMap, localMapping, loopClosing, frame, inliers, lastRelocFrameId, lastKeyFrameFrameId);
            }
            else
            {
                if (trackingFrameCount > maxTrackingFrameCount)
                    maxTrackingFrameCount = trackingFrameCount;
                trackingFrameCount = 0;
                isTracking == false;
            }
        }
        else
        {
            int inliers;
            if (WAISlam::relocalization(frame, map, localMap, inliers))
            {
                isTracking     = true;
                relocalizeOnce = true;

                WAISlamTools::motionModel(frame, lastFrame, velocity, extrinsic);
                WAISlamTools::serialMapping(map, localMap, localMapping, loopClosing, frame, inliers, lastRelocFrameId, lastKeyFrameFrameId);
            }
        }

        lastFrame = WAIFrame(frame);
        frameCount++;
    }

    if (trackingFrameCount > maxTrackingFrameCount)
        maxTrackingFrameCount = trackingFrameCount;

    result.frameCount         = frameCount;
    result.trackingFrameCount = maxTrackingFrameCount;
    result.ratio              = ((float)maxTrackingFrameCount / (float)frameCount);
    result.wasSuccessful      = relocalizeOnce;

    delete (localMapping);
    delete (loopClosing);

    return result;
}

bool TestRunnerView::loadSites(const std::string&         erlebARDir,
                               const std::string&         configFile,
                               const std::string&         calibrationsDir,
                               std::vector<TestInstance>& testInstances)
{
    Utils::log("debug", "TestRunnerView::loadSites");
    //parse config file
    cv::FileStorage fs;
    Utils::log("debug", "TestRunnerView::loadSites: erlebBarDir %s", erlebARDir.c_str());
    Utils::log("debug", "TestRunnerView::loadSites: configFile %s", configFile.c_str());

    fs.open(configFile, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        Utils::log("Error", "TestRunner::loadSites: Could not open configFile: %s", configFile.c_str());
        return false;
    }

    //helper for areas that have been enabled
    //std::set<std::string> enabledAreas;

    //std::string erlebARDirUnified = Utils::unifySlashes(erlebARDir);

    //setup for enabled areas
    cv::FileNode locsNode = fs["locations"];
    for (auto itLocs = locsNode.begin(); itLocs != locsNode.end(); itLocs++)
    {
        cv::FileNode areasNode = (*itLocs)["areas"];
        for (auto itAreas = areasNode.begin(); itAreas != areasNode.end(); itAreas++)
        {
            cv::FileNode videosNode = (*itAreas)["videos"];
            for (auto itVideos = videosNode.begin(); itVideos != videosNode.end(); itVideos++)
            {
                std::string location = (*itLocs)["location"];
                std::string area     = (*itAreas)["area"];
                std::string map      = (*itAreas)["map"];
                std::string video    = *itVideos;

                TestInstance testInstance;
                testInstance.location = location;
                testInstance.area     = area;
                testInstance.map      = map;
                testInstance.video    = video;

                SlamVideoInfos slamVideoInfos;
                if (!extractSlamVideoInfosFromFileName(video, &slamVideoInfos))
                {
                    Utils::log("Error", "TestRunner::loadSites: Could not extract slam video infos: %s", video.c_str());
                    return false;
                }

                testInstance.calibration = "camCalib_" + slamVideoInfos.deviceString + "_main.xml";

                testInstances.push_back(testInstance);
            }
        }
    }

    return true;
}