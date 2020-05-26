#include <iostream>
#include <memory>

#include <Utils.h>

#include <WAICalibration.h>
#include <WAIMapStorage.h>

#include <WAIFrame.h>
#include <WAISlam.h>
#include <WAIKeyFrameDB.h>
#include <Utils.h>
#include <SENSVideoStream.h>

#include <FeatureExtractorFactory.h>
#include <ORBextractor.h>
#include <SURFextractor.h>
#include <GLSLextractor.h>

#include <WAITester.h>
#include <GLFW/glfw3.h>

#define TRACKING_FLAG 1
#define RELOC_FLAG 2

struct Config
{
    std::string   erlebARDir;
    std::string   configFile;
    std::string   vocFile;
    int           testFlags;
    int           frameRate;
    ExtractorType extractorType;
};

struct SlamVideoInfos
{
    std::string dateTime;
    std::string weatherConditions;
    std::string deviceString;
    std::string purpose;
    std::string resolution;
};

static bool extractSlamVideoInfosFromFileName(std::string     fileName,
                                              SlamVideoInfos* slamVideoInfos)
{
    bool result = false;

    std::vector<std::string> stringParts;
    Utils::splitString(fileName, '_', stringParts);

    if (stringParts.size() == 5)
    {
        slamVideoInfos->dateTime          = stringParts[0];
        slamVideoInfos->weatherConditions = stringParts[1];
        slamVideoInfos->deviceString      = stringParts[2];
        slamVideoInfos->purpose           = stringParts[3];
        slamVideoInfos->resolution        = stringParts[4];

        result = true;
    }

    return result;
}

Tester::RelocalizationTestResult Tester::runRelocalizationTest(std::string       videoFile,
                                                               std::string       mapFile,
                                                               WAIOrbVocabulary* voc,
                                                               CVCalibration&    calibration,
                                                               ExtractorType     extractorType)
{
    RelocalizationTestResult result = {};

    //TODO FIX NOW
    // TODO(dgj1): this is kind of a hack... improve (maybe separate function call??)
    WAIFrame::mbInitialComputations = true;

    WAIKeyFrameDB* keyFrameDB = new WAIKeyFrameDB(voc);

    WAIMap* map = new WAIMap(keyFrameDB);
    WAIMapStorage::loadMap(map, nullptr, _voc, mapFile, false, true);

    SENSVideoStream vstream(videoFile, false, false, false);

    CVSize2i                     videoSize       = vstream.getFrameSize();
    float                        widthOverHeight = (float)videoSize.width / (float)videoSize.height;
    std::unique_ptr<KPextractor> extractor       = _factory.make(extractorType, {videoSize.width, videoSize.height});

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
                                         voc,
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

Tester::TrackingTestResult Tester::runTrackingTest(std::string       videoFile,
                                                   std::string       mapFile,
                                                   WAIOrbVocabulary* voc,
                                                   CVCalibration&    calibration,
                                                   ExtractorType     extractorType,
                                                   int               framerate)
{
    TrackingTestResult result = {};

    WAIFrame::mbInitialComputations = true;

    WAIKeyFrameDB* keyFrameDB = new WAIKeyFrameDB(voc);

    WAIMap* map = new WAIMap(keyFrameDB);
    WAIMapStorage::loadMap(map, nullptr, voc, mapFile, false, true);

    LocalMapping* localMapping = new ORB_SLAM2::LocalMapping(map, 1.0f, voc, 0.95f);
    LoopClosing*  loopClosing  = new ORB_SLAM2::LoopClosing(map, voc, false, false);

    localMapping->SetLoopCloser(loopClosing);
    loopClosing->SetLocalMapper(localMapping);

    SENSVideoStream              vstream(videoFile, false, false, false, (float)framerate);
    CVSize2i                     videoSize       = vstream.getFrameSize();
    float                        widthOverHeight = (float)videoSize.width / (float)videoSize.height;
    std::unique_ptr<KPextractor> extractor       = _factory.make(extractorType, {videoSize.width, videoSize.height});

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

void printHelp()
{
    std::stringstream ss;
    ss << "app-Test-WAI for testing relocalization or tracking!" << std::endl;
    ss << "Example1 (win):  app-Test-WAI.exe -erlebARDir C:/Erleb-AR -configFile testConfig.json" << std::endl;
    ss << "Example2 (unix): ./app-Test-WAI -erlebARDir ~/Erleb-AR -configFile testConfig.json" << std::endl;
    ss << "" << std::endl;
    ss << "Options: " << std::endl;
    ss << "  -h/-help        print this help, e.g. -h" << std::endl;
    ss << "  -erlebARDir     Path to Erleb-AR root directory" << std::endl;
    ss << "  -configFile     Path and name to TestConfig.json" << std::endl;
    ss << "  -vocFile        Path and name to Vocabulary file" << std::endl;
    ss << "  -f [fps]        Specify number of frames per seconds" << std::endl;
    ss << "  -t              test tracking only" << std::endl;
    ss << "  -r              test relocalization only" << std::endl;
    ss << "  -rt             test tracking and relocalization" << std::endl;

    std::cout << ss.str() << std::endl;
}

void readArgs(int argc, char* argv[], Config& config)
{
    config.testFlags     = TRACKING_FLAG;
    config.frameRate     = 0;
    config.extractorType = ExtractorType_GLSL_1;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "-erlebARDir") && i + 1 < argc)
        {
            config.erlebARDir = argv[++i];
        }
        else if (!strcmp(argv[i], "-configFile") && i + 1 < argc)
        {
            config.configFile = argv[++i];
        }
        else if (!strcmp(argv[i], "-vocFile") && i + 1 < argc)
        {
            config.vocFile = argv[++i];
        }
        else if (!strcmp(argv[i], "-rt"))
        {
            config.testFlags = TRACKING_FLAG | RELOC_FLAG;
        }
        else if (!strcmp(argv[i], "-r"))
        {
            config.testFlags = RELOC_FLAG;
        }
        else if (!strcmp(argv[i], "-f") && i + 1 < argc)
        {
            config.frameRate = atoi(argv[i + 1]);
            i++;
            std::cout << "framerate " << config.frameRate << std::endl;
        }
        else if (!strcmp(argv[i], "-feature"))
        {
            i++;
            if (!strcmp(argv[i], "SURF_BRIEF_500"))
                config.extractorType = ExtractorType_SURF_BRIEF_500;
            else if (!strcmp(argv[i], "SURF_BRIEF_800"))
                config.extractorType = ExtractorType_SURF_BRIEF_800;
            else if (!strcmp(argv[i], "SURF_BRIEF_1000"))
                config.extractorType = ExtractorType_SURF_BRIEF_1000;
            else if (!strcmp(argv[i], "SURF_BRIEF_1200"))
                config.extractorType = ExtractorType_SURF_BRIEF_1200;
            else if (!strcmp(argv[i], "FAST_ORBS_1000"))
                config.extractorType = ExtractorType_FAST_ORBS_1000;
            else if (!strcmp(argv[i], "FAST_ORBS_2000"))
                config.extractorType = ExtractorType_FAST_ORBS_2000;
            else if (!strcmp(argv[i], "FAST_ORBS_4000"))
                config.extractorType = ExtractorType_FAST_ORBS_4000;
            else if (!strcmp(argv[i], "GLSL_1"))
                config.extractorType = ExtractorType_GLSL_1;
            else if (!strcmp(argv[i], "GLSL"))
                config.extractorType = ExtractorType_GLSL;
        }
        else
        {
            printHelp();
        }
    }
}

void Tester::loadSites(const std::string& erlebARDir, const std::string& configFile)
{
    try
    {
        WAI_DEBUG("Tester: loading sites:");
        //parse config file
        cv::FileStorage fs;
        std::cout << "erlebBarDir " << erlebARDir << std::endl;
        std::cout << "configFile " << configFile << std::endl
                  << std::endl;

        fs.open(configFile, cv::FileStorage::READ);
        if (!fs.isOpened())
            throw std::runtime_error("Tester::loadSites: Could not open configFile: " + configFile);

        //helper for areas that have been enabled
        std::set<std::string> enabledAreas;

        std::string erlebARDirUnified = Utils::unifySlashes(erlebARDir);

        //setup for enabled areas
        cv::FileNode locsNode = fs["locationsEnabling"];
        for (auto itLocs = locsNode.begin(); itLocs != locsNode.end(); ++itLocs)
        {
            std::string  location  = (*itLocs)["location"];
            cv::FileNode areasNode = (*itLocs)["areas"];
            Areas        areas;
            for (auto itAreas = areasNode.begin(); itAreas != areasNode.end(); ++itAreas)
            {
                Area area    = (*itAreas)["area"];
                bool enabled = false;
                (*itAreas)["enabled"] >> enabled;
                if (enabled)
                {
                    WAI_DEBUG("Tester::loadSites: enabling %s %s", location.c_str(), area.c_str());
                    Areas& areas = _erlebAR[location];
                    Datas  datas = Datas();

                    //insert empty Videos vector
                    areas.insert(std::pair<Area, Datas>(area, datas));
                    enabledAreas.insert(area);
                }
            }
        }

        //try to find corresponding files in sites directory and add full file paths to _sites
        cv::FileNode videoAreasNode = fs["mappingVideos"];
        for (auto itVideoAreas = videoAreasNode.begin(); itVideoAreas != videoAreasNode.end(); ++itVideoAreas)
        {
            Location     location  = (*itVideoAreas)["location"];
            cv::FileNode areasNode = (*itVideoAreas)["area"];
            Area         area;
            std::string  map;
            std::string  mapFile;
            areasNode["name"] >> area;
            areasNode["map"] >> map;
            mapFile = erlebARDirUnified + "locations/" + location + "/" + area + "/" + "maps/" + map;
            if (!Utils::fileExists(mapFile))
                throw std::runtime_error("Tester::loadSites: Map file does not exist: " + mapFile);

            if (enabledAreas.find(area) != enabledAreas.end())
            {
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
                    if (!Utils::fileExists(_calibrationsDir + calibFile))
                        throw std::runtime_error("Tester::loadSites: Calibration file does not exist: " + _calibrationsDir + calibFile);

                    //load calibration file and check for aspect ratio
                    if (!testData.calibration.load(_calibrationsDir, calibFile, true))
                        throw std::runtime_error("Tester::loadSites: Could not load calibration file: " + _calibrationsDir + calibFile);

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
                    _erlebAR[location][area].push_back(testData);
                }
            }
        }
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

static GLFWwindow* window;          //!< The global glfw window handle
static SLint       svIndex;         //!< SceneView index
static SLint       scrWidth  = 640; //!< Window width at start up
static SLint       scrHeight = 480; //!< Window height at start up
//-----------------------------------------------------------------------------
/*!
Error callback handler for GLFW.
*/
void onGLFWError(int error, const char* description)
{
    fputs(description, stderr);
}

void GLFWInit()
{
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(onGLFWError);

    // Enable fullscreen anti aliasing with 4 samples
    glfwWindowHint(GLFW_SAMPLES, 4);

    //You can enable or restrict newer OpenGL context here (read the GLFW documentation)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(scrWidth, scrHeight, "WAI Demo", nullptr, nullptr);

    //get real window size
    glfwGetWindowSize(window, &scrWidth, &scrHeight);

    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Get the current GL context. After this you can call GL
    glfwMakeContextCurrent(window);

    // Init OpenGL access library gl3w
    if (gl3wInit()!=0)
    {
        cerr << "Failed to initialize OpenGL" << endl;
        exit(-1);
    }
}

Tester::~Tester()
{
}

Tester::Tester(std::string erlebARDir, std::string configFile, std::string vocFile, int testFlags, int frameRate, ExtractorType extractorType)
  : _erlebARDir(Utils::unifySlashes(erlebARDir))
{
    _calibrationsDir = _erlebARDir + "calibrations/";
    _testFlags       = testFlags;
    _framerate       = frameRate;
    _extractorType   = extractorType;
    _voc = new WAIOrbVocabulary();
    _voc->loadFromFile(vocFile);

    //scan erlebar directory and config file, collect everything that is enabled in the config file and
    //check that all files (video and calibration) exist.
    loadSites(erlebARDir, configFile);
}

void Tester::launchRelocalizationTest(const Location& location, const Area& area, Datas& datas, ExtractorType extractorType)
{
    WAI_INFO("Tester::lauchTest: Starting relocalization test for area: %s", area.c_str());
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
            RelocalizationTestResult r = runRelocalizationTest(testData.videoFile, testData.mapFile, _voc, testData.calibration, extractorType);

            printf("%s;%s;%s;%i;%i;%.2f\n",
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
        WAI_WARN("Tester::launchRelocalizationTest: No relocalization test for area: %s", area.c_str());
    }

    WAI_INFO("Tester::launchRelocalizationTest: Finished relocalization test for area: %s", area.c_str());
}

void Tester::launchTrackingTest(const Location& location, const Area& area, Datas& datas, ExtractorType extractorType, int framerate)
{
    WAI_INFO("Tester::lauchTest: Starting tracking test for area: %s", area.c_str());
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
            TrackingTestResult r = runTrackingTest(testData.videoFile, testData.mapFile, _voc, testData.calibration, extractorType, _framerate);

            if (r.wasSuccessful)
            {
                printf("%s;%s;%s;%i;%i;%.2f\n",
                       location.c_str(),
                       testData.videoFile.c_str(),
                       testData.mapFile.c_str(),
                       r.frameCount,
                       r.trackingFrameCount,
                       r.ratio);
            }
            else
            {
                printf("Never able to start traking\n");
            }
        }
    }
    else
    {
        WAI_WARN("Tester::launchTrackingTest: No tracking test for area: %s", area.c_str());
    }

    WAI_INFO("Tester::launchTrackingTest: Finished tracking test for area: %s", area.c_str());
}

void Tester::execute()
{
    try
    {
        for (auto itLocations = _erlebAR.begin(); itLocations != _erlebAR.end(); ++itLocations)
        {
            Areas& areas = itLocations->second;
            for (auto itAreas = areas.begin(); itAreas != areas.end(); ++itAreas)
            {
                if (_testFlags & RELOC_FLAG)
                    launchRelocalizationTest(itLocations->first, itAreas->first, itAreas->second, _extractorType);
                if (_testFlags & TRACKING_FLAG)
                    launchTrackingTest(itLocations->first, itAreas->first, itAreas->second, _extractorType, _framerate);
            }
        }
    }
    catch (std::exception& e)
    {
        throw std::runtime_error("Exception in Tester::execute: " + std::string(e.what()));
    }
    catch (...)
    {
        throw std::runtime_error("Unknown exception catched in Tester::execute!");
    }
}

int main(int argc, char* argv[])
{
    GLFWInit();
    try
    {
        Config config;

        //parse arguments
        readArgs(argc, argv, config);

        //initialize logger
        std::string cwd = Utils::getCurrentWorkingDir();
        Utils::initFileLog(Utils::unifySlashes(config.erlebARDir) + "Tester/", true);
        Utils::log("Main", "Tester");

        //init wai tester
        DUtils::Random::SeedRandOnce(1337);
        Tester tester(config.erlebARDir, config.configFile, config.vocFile, config.testFlags, config.frameRate, config.extractorType);
        tester.execute();
    }
    catch (std::exception& e)
    {
        Utils::log("Main", "Exception catched during test: %s", e.what());
    }
    catch (...)
    {
        Utils::log("Main", "Unknown exception during test");
    }

    return 0;
}
