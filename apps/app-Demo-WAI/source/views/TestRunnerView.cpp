#include <views/TestRunnerView.h>
#include <SENSCamera.h>
#include <WAIMapStorage.h>
#include <AppWAISlamParamHelper.h>
#include <Utils.h>
#include <FtpUtils.h>

TestRunnerView::TestRunnerView(sm::EventHandler&   eventHandler,
                               SLInputManager&     inputManager,
                               const ImGuiEngine&  imGuiEngine,
                               ErlebAR::Resources& resources,
                               const DeviceData&   deviceData)
  : SLSceneView(&_scene, deviceData.dpi(), inputManager),
    _gui(imGuiEngine, eventHandler, resources, deviceData.dpi(), deviceData.fontDir()),
    _scene("TestRunnerScene", nullptr),
    _testMode(TestMode_None),
    _erlebARDir(deviceData.erlebARTestDir()),
    _calibDir(deviceData.erlebARCalibTestDir()),
    _localMapping(nullptr),
    _loopClosing(nullptr),
    _ftpHost("pallas.bfh.ch:21"),
    _ftpUser("upload"),
    _ftpPwd("FaAdbD3F2a"),
    _ftpDir("erleb-AR/"),
    _videoWasDownloaded(false),
    _summedTime(0.0f)
{
    init("TestRunnerView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, &_gui, deviceData.writableDir());

#if USE_FBOW
    std::string vocabularyFile = "voc_fbow.bin";
#else
    std::string vocabularyFile = "ORBvoc.bin";
#endif

    _voc.loadFromFile(deviceData.vocabularyDir() + vocabularyFile);

    std::string configPath = "TestRunner/config/";
    std::string configDir  = _erlebARDir + configPath;
    Utils::makeDirRecurse(configDir);

    std::string errorMsg;
    FtpUtils::downloadAllFilesFromDir(configDir,
                                      _ftpHost,
                                      _ftpUser,
                                      _ftpPwd,
                                      _ftpDir + configPath,
                                      "json",
                                      errorMsg);

    _configFiles = Utils::getAllNamesInDir(configDir);

    onInitialize();
}

bool TestRunnerView::start(TestMode testMode, std::string configFile)
{
    bool result = false;

    if (loadSites(_erlebARDir, configFile, _calibDir, _testInstances))
    {
        _testMode         = testMode;
        _currentTestIndex = 0;

        _testStarted = true;

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

            HighResTimer timer = HighResTimer();

            WAIFrame currentFrame = WAIFrame(sensFrame.get()->imgManip,
                                             0.0f,
                                             _extractor.get(),
                                             intrinsic,
                                             distortion,
                                             &_voc,
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
                            WAISlamTools::mapping(_map, _localMap, _localMapping, currentFrame, _inliers, _lastRelocFrameId, _lastKeyFrameFrameId);
                        }
                        else
                        {
                            if (_trackingFrameCount > _maxTrackingFrameCount)
                            {
                                _maxTrackingFrameCount = _trackingFrameCount;
                            }
                            _trackingFrameCount = 0;
                            _isTracking         = false;
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
                            WAISlamTools::mapping(_map, _localMap, _localMapping, currentFrame, _inliers, _lastRelocFrameId, _lastKeyFrameFrameId);
                        }
                    }

                    _lastFrame = WAIFrame(currentFrame);
                }
                break;
            }

            timer.stop();
            _summedTime += timer.elapsedTimeInMilliSec();

            _currentFrameIndex++;
        }
        else
        {
            if (_vStream)
            {
                TestRunnerView::TestInstance currentTest = _testInstances[_currentTestIndex];

                float avgTime = _summedTime / (float)_currentFrameIndex;

                switch (_testMode)
                {
                    case TestMode_Relocalization: {
                        _testResults +=
                          currentTest.location + ";" +
                          currentTest.area + ";" +
                          currentTest.video + ";" +
                          currentTest.map + ";" +
                          currentTest.extractorType + ";" +
                          std::to_string(currentTest.nLevels) + ";" +
                          std::to_string(avgTime) + ";" +
                          std::to_string(_currentFrameIndex) + ";" +
                          std::to_string(_relocalizationFrameCount) + ";" +
                          Utils::toString((float)_relocalizationFrameCount / (float)_currentFrameIndex, 2) + "\n";
                    }
                    break;

                    case TestMode_Tracking: {
                        if (_trackingFrameCount > _maxTrackingFrameCount)
                        {
                            _maxTrackingFrameCount = _trackingFrameCount;
                        }

                        _testResults +=
                          currentTest.location + ";" +
                          currentTest.area + ";" +
                          currentTest.video + ";" +
                          currentTest.map + ";" +
                          currentTest.extractorType + ";" +
                          std::to_string(currentTest.nLevels) + ";" +
                          std::to_string(avgTime) + ";" +
                          std::to_string(_currentFrameIndex) + ";" +
                          std::to_string(_maxTrackingFrameCount) + ";" +
                          Utils::toString((float)_maxTrackingFrameCount / (float)_currentFrameIndex, 2) + "\n";

                        _maxTrackingFrameCount = 0;

                        _localMapping->RequestFinish();
                        _loopClosing->RequestFinish();

                        _mappingThread->join();
                        _loopClosingThread->join();
                    }
                    break;
                }

                if (_videoWasDownloaded)
                {
                    TestRunnerView::TestInstance testInstance    = _testInstances[_currentTestIndex];
                    std::string                  testInstanceDir = "locations/" +
                                                  testInstance.location + "/" +
                                                  testInstance.area + "/";

                    std::string videoDir  = testInstanceDir + "videos/";
                    std::string videoFile = _erlebARDir + videoDir + testInstance.video;
                    if (fileExists(videoFile))
                    {
                        Utils::deleteFile(videoFile);
                    }

                    _videoWasDownloaded = false;
                }

                _currentTestIndex++;
            }

            if (_currentTestIndex == _testInstances.size())
            {
                // done with current tests
                _testStarted = false;

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

                f << _testResults;

                f.flush();
                f.close();

                _testResults = "";

                // upload results to pallas

                std::string errorMsg;
                if (!FtpUtils::uploadFile(resultDir,
                                          resultFileName,
                                          _ftpHost,
                                          _ftpUser,
                                          _ftpPwd,
                                          _ftpDir + "TestRunner/",
                                          errorMsg))
                {
                    Utils::log("WAI", "TestRunner::update: Could not upload results file to pallas %s", errorMsg.c_str());
                }

                _testInstances.clear();
            }
            else
            {
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
                                                    _ftpHost,
                                                    _ftpUser,
                                                    _ftpPwd,
                                                    _ftpDir + mapDir,
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
                                                    _ftpHost,
                                                    _ftpUser,
                                                    _ftpPwd,
                                                    _ftpDir + videoDir,
                                                    errorMsg))
                        {
                            Utils::log("WAI", "TestRunner::loadSites: Failed to load video file %s: %s", testInstance.video.c_str(), errorMsg.c_str());
                            _currentTestIndex++;
                            continue;
                        }

                        _videoWasDownloaded = true;
                    }

                    std::string calibrationFile = _calibDir + testInstance.calibration;
                    if (!Utils::fileExists(calibrationFile))
                    {
                        std::string errorMsg;
                        if (!FtpUtils::downloadFile(_calibDir,
                                                    testInstance.calibration,
                                                    _ftpHost,
                                                    _ftpUser,
                                                    _ftpPwd,
                                                    _ftpDir + "calibrations/",
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

                    WAIKeyFrameDB* keyFrameDB = new WAIKeyFrameDB(&_voc);

                    if (_map)
                        delete _map;

                    _map = new WAIMap(keyFrameDB);
                    WAIMapStorage::loadMap(_map, nullptr, &_voc, mapFile, false, true);

                    if (_localMapping)
                        delete _localMapping;
                    if (_loopClosing)
                        delete _loopClosing;

                    if (_testMode == TestMode_Tracking)
                    {
                        _lastKeyFrameFrameId = 0;
                        _lastRelocFrameId    = 0;
                        _inliers             = 0;

                        _isTracking     = false;
                        _relocalizeOnce = false;

                        _localMap.keyFrames.clear();
                        _localMap.mapPoints.clear();
                        _localMap.refKF = nullptr;

                        _localMapping = new ORB_SLAM2::LocalMapping(_map, &_voc, 0.95f);
                        _loopClosing  = new ORB_SLAM2::LoopClosing(_map, &_voc, false, false);

                        _localMapping->SetLoopCloser(_loopClosing);
                        _loopClosing->SetLocalMapper(_localMapping);

                        _mappingThread     = new std::thread(&LocalMapping::Run, _localMapping);
                        _loopClosingThread = new std::thread(&LoopClosing::Run, _loopClosing);
                    }

                    if (_vStream)
                        delete _vStream;

                    _vStream = new SENSVideoStream(videoFile, false, false, false);

                    SENSFramePtr sensFrame = _vStream->grabNextFrame();

                    _extractor = _featureExtractorFactory.make(testInstance.extractorType, {sensFrame->captureWidth, sensFrame->captureHeight}, testInstance.nLevels);
                    if (!_extractor)
                    {
                        Utils::log("WAI", "TestRunner::loadSites: Could not create feature extractor with type: %s", testInstance.extractorType.c_str());
                        _currentTestIndex++;
                        continue;
                    }

                    _frameCount = 1;

                    while ((sensFrame = _vStream->grabNextFrame()))
                    {
                        _frameCount++;
                    }

                    // TODO(dgj1): this restarts the video, as setting the prop in android didn't work...
                    delete _vStream;
                    _vStream = new SENSVideoStream(videoFile, false, false, false);

                    _currentFrameIndex        = 0;
                    _relocalizationFrameCount = 0;
                    _trackingFrameCount       = 0;
                    _maxTrackingFrameCount    = 0;
                    _lastFrame                = WAIFrame();
                    _summedTime               = 0.0f;

                    instanceFound = true;
                }
            }
        }
    }

    bool result = onPaint();
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

    //setup for enabled areas
    cv::FileNode locsNode = fs["locations"];
    for (auto itLocs = locsNode.begin(); itLocs != locsNode.end(); itLocs++)
    {
        cv::FileNode areasNode = (*itLocs)["areas"];
        for (auto itAreas = areasNode.begin(); itAreas != areasNode.end(); itAreas++)
        {
            cv::FileNode mapsNode = (*itAreas)["maps"];
            for (auto itMaps = mapsNode.begin(); itMaps != mapsNode.end(); itMaps++)
            {
                cv::FileNode videosNode = (*itAreas)["videos"];
                for (auto itVideos = videosNode.begin(); itVideos != videosNode.end(); itVideos++)
                {
                    std::string location = (*itLocs)["location"];
                    std::string area     = (*itAreas)["area"];
                    std::string map      = *itMaps;
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

                    SlamMapInfos slamMapInfos;
                    if (!extractSlamMapInfosFromFileName(map, &slamMapInfos))
                    {
                        Utils::log("WAI", "TestRunner::loadSites: Could not extract slam map infos: %s", map.c_str());
                        return false;
                    }

                    testInstance.extractorType = slamMapInfos.extractorType;
                    testInstance.nLevels       = slamMapInfos.nLevels;

                    testInstances.push_back(testInstance);
                }
            }
        }
    }

    return true;
}
