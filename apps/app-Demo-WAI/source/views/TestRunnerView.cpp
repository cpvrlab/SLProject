#include <views/TestRunnerView.h>
#include <SENSCamera.h>
#include <WAIMapStorage.h>
#include <AppWAISlamParamHelper.h>
#include <Utils.h>

TestRunnerView::TestRunnerView(sm::EventHandler& eventHandler,
                               SLInputManager&   inputManager,
                               int               screenWidth,
                               int               screenHeight,
                               int               dotsPerInch,
                               std::string       erlebARDir,
                               std::string       calibDir,
                               std::string       configFile,
                               std::string       vocabularyFile,
                               std::string       imguiIniPath)
  : SLSceneView(&_scene, dotsPerInch, inputManager),
    _gui(eventHandler),
    _scene("TestRunnerScene", nullptr),
    _testMode(TestMode_None),
    _erlebARDir(erlebARDir),
    _calibDir(calibDir),
    _configFile(configFile),
    _vocFile(vocabularyFile)
{
    init("TestRunnerView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    onInitialize();
}

bool TestRunnerView::start(TestMode      testMode,
                           ExtractorType extractorType)
{
    _erlebARTestSet      = loadSites(_erlebARDir, _configFile, _calibDir);
    _locationIterator    = _erlebARTestSet.begin();
    _locationIteratorEnd = _erlebARTestSet.end();
    _areaIterator        = _locationIterator->second.begin();
    _areaIteratorEnd     = _locationIterator->second.end();
    _testIterator        = _areaIterator->second.begin();
    _testIteratorEnd     = _areaIterator->second.end();

    _testsDone   = false;
    _testStarted = true;

    _extractorType = extractorType;

    return true;
}

bool TestRunnerView::update()
{
    if (_testStarted)
    {
        SENSFramePtr sensFrame;
        if (_vStream && (sensFrame = _vStream->grabNextFrame()))
        {
            cv::Mat intrinsic  = _testIterator->calibration.cameraMat();
            cv::Mat distortion = _testIterator->calibration.distortion();

            WAIFrame currentFrame = WAIFrame(sensFrame.get()->imgGray,
                                             0.0f,
                                             _extractor.get(),
                                             intrinsic,
                                             distortion,
                                             WAIOrbVocabulary::get(),
                                             false);

            int      inliers;
            LocalMap localMap;
            localMap.keyFrames.clear();
            localMap.mapPoints.clear();
            localMap.refKF = nullptr;
            if (WAISlam::relocalization(currentFrame, _map, localMap, inliers))
            {
                _relocalizationFrameCount++;
            }

            _currentFrameIndex++;

            if (_currentFrameIndex == _frameCount)
            {
                testResults +=
                  _locationIterator->first + ";" +
                  _testIterator->videoFile + ";" +
                  _testIterator->mapFile + ";" +
                  std::to_string(_frameCount) + ";" +
                  std::to_string(_relocalizationFrameCount) + ";" +
                  std::to_string(_relocalizationFrameCount / _frameCount) + "\n";

                _testIterator++;
            }
        }
        else
        {
            if (_testIterator == _testIteratorEnd)
            {
                _areaIterator++;

                if (_areaIterator == _areaIteratorEnd)
                {
                    _locationIterator++;

                    if (_locationIterator == _locationIteratorEnd)
                    {
                        // done with current tests
                        _testStarted = false;
                        _testsDone   = true;

                        std::string resultDir = _erlebARDir + "TestRunner/";
                        if (!Utils::dirExists(resultDir))
                            Utils::makeDir(resultDir);

                        // save results in file
                        std::string resultFile = resultDir + "results.csv";

                        std::ofstream f;
                        f.open(resultFile, std::ofstream::out);

                        if (!f.is_open())
                        {
                            throw std::runtime_error("Could not open file" + resultFile);
                        }

                        f << testResults;

                        f.flush();
                        f.close();
                    }
                    else
                    {
                        _areaIterator    = _locationIterator->second.begin();
                        _areaIteratorEnd = _locationIterator->second.end();
                        _testIterator    = _areaIterator->second.begin();
                        _testIteratorEnd = _areaIterator->second.end();
                    }
                }
                else
                {
                    _testIterator    = _areaIterator->second.begin();
                    _testIteratorEnd = _areaIterator->second.end();
                }
            }

            if (_testIterator != _testIteratorEnd)
            {
                WAIFrame::mbInitialComputations = true;

                WAIOrbVocabulary::initialize(_vocFile);
                ORBVocabulary* orbVoc     = WAIOrbVocabulary::get();
                WAIKeyFrameDB* keyFrameDB = new WAIKeyFrameDB(*orbVoc);

                _map = new WAIMap(keyFrameDB);
                WAIMapStorage::loadMap(_map, nullptr, orbVoc, _testIterator->mapFile, false, true);

                if (_vStream)
                    delete _vStream;

                _vStream = new SENSVideoStream(_testIterator->videoFile, false, false, false);

                CVSize2i videoSize       = _vStream->getFrameSize();
                float    widthOverHeight = (float)videoSize.width / (float)videoSize.height;
                _extractor               = _featureExtractorFactory.make(_extractorType, {videoSize.width, videoSize.height});

                _videoName                = _testIterator->videoFile;
                _frameCount               = _vStream->frameCount();
                _currentFrameIndex        = 0;
                _relocalizationFrameCount = 0;
            }
        }
    }

    bool result = onPaint();
    return result;
}

void TestRunnerView::launchRelocalizationTest(const Location&        location,
                                              const Area&            area,
                                              std::vector<TestData>& datas,
                                              ExtractorType          extractorType,
                                              std::string            vocabularyFile)
{
    Utils::log("info", "TestRunnerView::launchTest: Starting relocalization test for area: %s", area.c_str());
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
            RelocalizationTestResult r = runRelocalizationTest(testData.videoFile, testData.mapFile, vocabularyFile, testData.calibration, extractorType);

            Utils::log("warn",
                       "%s;%s;%s;%i;%i;%.2f\n",
                       location.c_str(),
                       testData.videoFile.c_str(),
                       testData.mapFile.c_str(),
                       r.frameCount,
                       r.relocalizationFrameCount,
                       r.ratio);
        }
    }
    else
    {
        Utils::log("warn", "TestRunnerView::launchRelocalizationTest: No relocalization test for area: %s", area.c_str());
    }

    Utils::log("info", "TestRunnerView::launchRelocalizationTest: Finished relocalization test for area: %s", area.c_str());
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

TestRunnerView::RelocalizationTestResult TestRunnerView::runRelocalizationTest(std::string    videoFile,
                                                                               std::string    mapFile,
                                                                               std::string    vocFile,
                                                                               CVCalibration& calibration,
                                                                               ExtractorType  extractorType)
{
    RelocalizationTestResult result = {};

    //TODO FIX NOW
    // TODO(dgj1): this is kind of a hack... improve (maybe separate function call??)
    WAIFrame::mbInitialComputations = true;

    WAIOrbVocabulary::initialize(vocFile);
    ORBVocabulary* orbVoc     = WAIOrbVocabulary::get();
    WAIKeyFrameDB* keyFrameDB = new WAIKeyFrameDB(*orbVoc);

    WAIMap* map = new WAIMap(keyFrameDB);
    WAIMapStorage::loadMap(map, nullptr, orbVoc, mapFile, false, true);

    SENSVideoStream vstream(videoFile, false, false, false);

    CVSize2i                     videoSize       = vstream.getFrameSize();
    float                        widthOverHeight = (float)videoSize.width / (float)videoSize.height;
    std::unique_ptr<KPextractor> extractor       = _featureExtractorFactory.make(extractorType, {videoSize.width, videoSize.height});

    unsigned int lastRelocFrameId         = 0;
    int          frameCount               = 0;
    int          relocalizationFrameCount = 0;
    while (SENSFramePtr sensFrame = vstream.grabNextFrame())
    {
        cv::Mat  intrinsic    = calibration.cameraMat();
        cv::Mat  distortion   = calibration.distortion();
        WAIFrame currentFrame = WAIFrame(sensFrame.get()->imgGray,
                                         0.0f,
                                         extractor.get(),
                                         intrinsic,
                                         distortion,
                                         orbVoc,
                                         false);

        int      inliers;
        LocalMap localMap;
        localMap.keyFrames.clear();
        localMap.mapPoints.clear();
        localMap.refKF = nullptr;
        if (WAISlam::relocalization(currentFrame, map, localMap, inliers))
        {
            relocalizationFrameCount++;
        }

        frameCount++;
    }

    result.frameCount               = frameCount;
    result.relocalizationFrameCount = relocalizationFrameCount;
    result.ratio                    = ((float)relocalizationFrameCount / (float)frameCount);
    result.wasSuccessful            = true;

    return result;
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

TestRunnerView::ErlebARTestSet TestRunnerView::loadSites(const std::string& erlebARDir,
                                                         const std::string& configFile,
                                                         const std::string& calibrationsDir)
{
    try
    {
        ErlebARTestSet result;

        Utils::log("debug", "TestRunnerView::loadSites");
        //parse config file
        cv::FileStorage fs;
        Utils::log("debug", "TestRunnerView::loadSites: erlebBarDir %s", erlebARDir.c_str());
        Utils::log("debug", "TestRunnerView::loadSites: configFile %s", configFile.c_str());

        fs.open(configFile, cv::FileStorage::READ);
        if (!fs.isOpened())
            throw std::runtime_error("TestRunnerView::loadSites: Could not open configFile: " + configFile);

        //helper for areas that have been enabled
        std::set<std::string> enabledAreas;

        std::string erlebARDirUnified = Utils::unifySlashes(erlebARDir);

        //setup for enabled areas
        cv::FileNode locsNode = fs["locationsEnabling"];
        for (auto itLocs = locsNode.begin(); itLocs != locsNode.end(); ++itLocs)
        {
            std::string  location  = (*itLocs)["location"];
            cv::FileNode areasNode = (*itLocs)["areas"];
            AreaMap      areas;
            for (auto itAreas = areasNode.begin(); itAreas != areasNode.end(); ++itAreas)
            {
                std::string area    = (*itAreas)["area"];
                bool        enabled = false;
                (*itAreas)["enabled"] >> enabled;
                if (enabled)
                {
                    Utils::log("debug", "Tester::loadSites: enabling %s %s", location.c_str(), area.c_str());
                    AreaMap&       areas = result[location];
                    TestDataVector datas = TestDataVector();

                    //insert empty Videos vector
                    areas.insert(std::pair<std::string, TestDataVector>(area, datas));
                    enabledAreas.insert(area);
                }
            }
        }

        //try to find corresponding files in sites directory and add full file paths to _sites
        cv::FileNode videoAreasNode = fs["mappingVideos"];
        for (auto itVideoAreas = videoAreasNode.begin(); itVideoAreas != videoAreasNode.end(); ++itVideoAreas)
        {
            std::string  location  = (*itVideoAreas)["location"];
            cv::FileNode areasNode = (*itVideoAreas)["area"];
            std::string  area;
            std::string  map;
            std::string  mapFile;
            areasNode["name"] >> area;

            if (enabledAreas.find(area) != enabledAreas.end())
            {
                areasNode["map"] >> map;
                mapFile = erlebARDirUnified + "locations/" + location + "/" + area + "/" + "maps/" + map;
                if (!Utils::fileExists(mapFile))
                    throw std::runtime_error("Tester::loadSites: Map file does not exist: " + mapFile);

                cv::FileNode videosNode = (*itVideoAreas)["videos"];
                for (auto itVideos = videosNode.begin(); itVideos != videosNode.end(); ++itVideos)
                {
                    //check if this is enabled
                    std::string name = *itVideos;
                    TestData    testData;
                    testData.videoFile = erlebARDirUnified + "locations/" + location + "/" + area + "/" + "videos/" + name;
                    testData.mapFile   = mapFile;

                    if (!Utils::fileExists(testData.videoFile))
                        throw std::runtime_error("Tester::loadSites: Video file does not exist: " + testData.videoFile);

                    //check if calibration file exists
                    SlamVideoInfos slamVideoInfos;

                    if (!extractSlamVideoInfosFromFileName(name, &slamVideoInfos))
                        throw std::runtime_error("Tester::loadSites: Could not extract slam video infos: " + name);

                    // construct calibrations file name and check if it exists
                    std::string calibFile = "camCalib_" + slamVideoInfos.deviceString + "_main.xml";

                    //videoAndCalib.calibFile = erlebARDirUnified + "../calibrations/" + "camCalib_" + slamVideoInfos.deviceString + "_main.xml";
                    if (!Utils::fileExists(calibrationsDir + calibFile))
                        throw std::runtime_error("Tester::loadSites: Calibration file does not exist: " + calibrationsDir + calibFile);

                    //load calibration file and check for aspect ratio
                    if (!testData.calibration.load(calibrationsDir, calibFile, true))
                        throw std::runtime_error("Tester::loadSites: Could not load calibration file: " + calibrationsDir + calibFile);

                    std::vector<std::string> size;
                    Utils::splitString(slamVideoInfos.resolution, 'x', size);
                    if (size.size() == 2)
                    {
                        int width  = std::stoi(size[0]);
                        int height = std::stoi(size[1]);
                        if (testData.calibration.imageSize().width != width ||
                            testData.calibration.imageSize().height != height)
                        {
                            testData.calibration.adaptForNewResolution(CVSize(width, height), true);

                            //throw std::runtime_error("Resolutions of video and calibration do not fit together. Using: " + calibFile + " and " + name);
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Tester::loadSites: Could not estimate resolution string: " + calibFile);
                    }

                    //add video to videos vector
                    result[location][area].push_back(testData);
                }
            }
        }

        return result;
    }
    catch (std::exception& e)
    {
        throw std::runtime_error("Exception in Tester::loadSites: " + std::string(e.what()));
    }
    catch (...)
    {
        throw std::runtime_error("Unknown exception catched in Tester::loadSites!");
    }
}