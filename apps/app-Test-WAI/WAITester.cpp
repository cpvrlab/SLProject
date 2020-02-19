#include <iostream>
#include <memory>

#include <Utils.h>
#include <CVCapture.h>

#include <WAICalibration.h>
#include <WAIMapStorage.h>

#include <WAIFrame.h>
#include <WAISlam.h>
#include <WAIKeyFrameDB.h>
#include <Utils.h>

#include <FeatureExtractorFactory.h>
#include <ORBextractor.h>
#include <SURFextractor.h>
#include <GLSLextractor.h>

#include <GLFW/glfw3.h>
#include <WAITester.h>

struct Config
{
    std::string erlebARDir;
    std::string configFile;
    std::string vocFile;
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

struct RelocalizationTestResult
{
    bool  wasSuccessful;
    int   frameCount;
    int   relocalizationFrameCount;
    float ratio;
};

struct RelocalizationTestCase
{
    std::string videoFileName;
    std::string calibrationFileName;
};

struct RelocalizationTestBench
{
    std::string location;
    std::string weatherConditions;

    std::string mapFileName;
    std::string vocFileName;

    std::vector<RelocalizationTestCase>   testCases;
    std::vector<RelocalizationTestResult> testResults;
};

RelocalizationTestResult runRelocalizationTest(std::string videoFile,
                                               std::string mapFile,
                                               std::string vocFile,
                                               CVCalibration &calibration)
{
    RelocalizationTestResult result = {};

    //TODO FIX NOW
    // TODO(dgj1): this is kind of a hack... improve (maybe separate function call??)
    WAIFrame::mbInitialComputations = true;

    WAIOrbVocabulary::initialize(vocFile);
    ORB_SLAM2::ORBVocabulary* orbVoc     = WAIOrbVocabulary::get();
    ORB_SLAM2::KPextractor*   extractor  = new ORB_SLAM2::SURFextractor(800);
    WAIKeyFrameDB*            keyFrameDB = new WAIKeyFrameDB(*orbVoc);

    WAIMap* map = new WAIMap(keyFrameDB);
    WAIMapStorage::loadMap(map, nullptr, orbVoc, mapFile, false, true);

    CVCapture::instance()->videoType(VT_FILE);
    CVCapture::instance()->videoFilename = videoFile;
    CVCapture::instance()->videoLoops    = false;

    CVSize2i videoSize       = CVCapture::instance()->openFile();
    float    widthOverHeight = (float)videoSize.width / (float)videoSize.height;

    unsigned int lastRelocFrameId         = 0;
    int          frameCount               = 0;
    int          relocalizationFrameCount = 0;
    while (CVCapture::instance()->grabAndAdjustForSL(widthOverHeight))
    {
        cv::Mat intrinsic = calibration.cameraMat();
        cv::Mat distortion = calibration.distortion();
        WAIFrame currentFrame = WAIFrame(CVCapture::instance()->lastFrameGray,
                                         0.0f,
                                         extractor,
                                         intrinsic, 
                                         distortion,
                                         orbVoc,
                                         false);

        int inliers;
        LocalMap localMap;
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


void printHelp()
{
    std::stringstream ss;
    ss << "app-Test-WAI for creation of Erleb-AR maps!" << std::endl;
    ss << "Example1 (win):  app-MapCreator.exe -erlebARDir C:/Erleb-AR -configFile testConfig.json" << std::endl;
    ss << "Example2 (unix): ./app-MapCreator -erlebARDir C:/Erleb-AR -configFile testConfig.json" << std::endl;
    ss << "" << std::endl;
    ss << "Options: " << std::endl;
    ss << "  -h/-help        print this help, e.g. -h" << std::endl;
    ss << "  -erlebARDir     Path to Erleb-AR root directory" << std::endl;
    ss << "  -configFile     Path and name to MapCreatorConfig.json" << std::endl;
    ss << "  -vocFile        Path and name to Vocabulary file" << std::endl;

    std::cout << ss.str() << std::endl;
}

void readArgs(int argc, char* argv[], Config& config)
{
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "-erlebARDir"))
        {
            config.erlebARDir = argv[++i];
        }
        else if (!strcmp(argv[i], "-configFile"))
        {
            config.configFile = argv[++i];
        }
        else if (!strcmp(argv[i], "-vocFile"))
        {
            config.vocFile = argv[++i];
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
        std::cout << "configFile " << configFile << std::endl << std::endl;

        fs.open(configFile, cv::FileStorage::READ);
        if (!fs.isOpened())
            throw std::runtime_error("Tester::loadSites: Could not open configFile: " + configFile);

        //helper for areas that have been enabled
        std::set<Area> enabledAreas;

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
                std::string area    = (*itAreas)["area"];
                bool        enabled = false;
                (*itAreas)["enabled"] >> enabled;
                if (enabled)
                {
                    WAI_DEBUG("MapCreator::loadSites: enabling %s %s", location.c_str(), area.c_str());
                    Areas&     areas      = _erlebAR[location];
                    Datas datas = Datas();

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
            Location location = (*itVideoAreas)["location"];
            Area     area     = (*itVideoAreas)["area"];
            if (enabledAreas.find(area) != enabledAreas.end())
            {
                cv::FileNode videosNode = (*itVideoAreas)["videos"];
                for (auto itVideos = videosNode.begin(); itVideos != videosNode.end(); ++itVideos)
                {
                    //check if this is enabled
                    std::string   name = *itVideos;
                    TestData testData;
                    testData.videoFile = erlebARDirUnified + "locations/" + location + "/" + area + "/" + "videos/" + name;

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
        throw std::runtime_error("Exception in MapCreator::loadSites: " + std::string(e.what()));
    }
    catch (...)
    {
        throw std::runtime_error("Unknown exception catched in MapCreator::loadSites!");
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

    // Include OpenGL via GLEW (init must be after window creation)
    // The goal of the OpenGL Extension Wrangler Library (GLEW) is to assist C/C++
    // OpenGL developers with two tedious tasks: initializing and using extensions
    // and writing portable applications. GLEW provides an efficient run-time
    // mechanism to determine whether a certain extension is supported by the
    // driver or not. OpenGL core and extension functionality is exposed via a
    // single header file. Download GLEW at: http://glew.sourceforge.net/
    glewExperimental = GL_TRUE; // avoids a crash
    GLenum err       = glewInit();
    if (GLEW_OK != err)
    {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

Tester::~Tester()
{
}

Tester::Tester(std::string erlebARDir, std::string configFile, std::string vocFile)
  : _erlebARDir(Utils::unifySlashes(erlebARDir))
{
    _calibrationsDir = _erlebARDir + "calibrations/";
    _vocFile         = vocFile;

    //scan erlebar directory and config file, collect everything that is enabled in the config file and
    //check that all files (video and calibration) exist.
    loadSites(erlebARDir, configFile);

    //init keypoint extractors
    //TODO(lulu) create extractor depending on video resolution especially if different for each video!
    FeatureExtractorFactory factory;
    _kpExtractor = factory.make(7, {640, 320});
}

void Tester::launchRelocalizationTest(const Location& location, const Area& area, Datas& datas)
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
            RelocalizationTestResult r = runRelocalizationTest(testData.videoFile, testData.map, _vocFile, testData.calibration);

            printf("%s;%s;%s;%s;%s;%s;%i;%i;%.2f\n",
                   location.c_str(),
                   b.weatherConditions.c_str(),
                   c.videoFileName.c_str(),
                   c.calibrationFileName.c_str(),
                   b.mapFileName.c_str(),
                   b.vocFileName.c_str(),
                   r.frameCount,
                   r.relocalizationFrameCount,
                   r.ratio);
        }
    }
    else
    {
        WAI_WARN("MapCreator::createNewWaiMap: No map created for area: %s", area.c_str());
    }

    WAI_INFO("MapCreator::createNewWaiMap: Finished map creation for area: %s", area.c_str());
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
                launchRelocalizationTest(itLocations->first, itAreas->first, itAreas->second);
            }
        }
    }
    catch (std::exception& e)
    {
        throw std::runtime_error("Exception in MapCreator::execute: " + std::string(e.what()));
    }
    catch (...)
    {
        throw std::runtime_error("Unknown exception catched in MapCreator::execute!");
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
        Utils::initFileLog(Utils::unifySlashes(config.erlebARDir) + "MapCreator/", true);
        Utils::log("Main", "MapCreator");

        //init map creator
        Tester Tester(config.erlebARDir, config.configFile, config.vocFile);
        //todo: call different executes e.g. executeFullProcessing(), executeThinOut()
        //input and output directories have to be defined together with json file which is always scanned during construction
    }
    catch (std::exception& e)
    {
        Utils::log("Main", "Exception catched during map creation: %s", e.what());
    }
    catch (...)
    {
        Utils::log("Main", "Unknown exception during map creation!");
    }

    DUtils::Random::SeedRandOnce(1337);

    std::vector<RelocalizationTestBench> testBenches;
    /*
    addRelocalizationTestCase(southwallBench, "160919-143001_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143001_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143002_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143003_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143004_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143001_test-whe505af1e71561618241641-2786283903-9c5cl-CLT-AL01_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "200919-154459_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");

    //testBenches.push_back(southwallBench);

/*
    RelocalizationTestBench southwallBench2 = createRelocalizationTestBench("southwall",
                                                                            "shade",
                                                                            "160919-143001_android-mcrd1-35-ASUS-A002.json",
                                                                            "voc_southwall_200919_154459_android-mcrd1-35-ASUS-A002_640.bin");

    addRelocalizationTestCase(southwallBench2, "160919-143001_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143003_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143004_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143001_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143002_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143003_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143004_cm-cm-build-c25-TA-1021_640.mp4");
    //addRelocalizationTestCase(southwallBench2, "160919-143001_test-whe505af1e71561618241641-2786283903-9c5cl-CLT-AL01_640.mp4");
    addRelocalizationTestCase(southwallBench2, "200919-154459_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "011019-164748_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "011019-164844_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "011019-165120_cm-cm-build-c25-TA-1021_640.mp4");

    addRelocalizationTestCase(southwallBench2, "160919-143001_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143001_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143002_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143003_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143004_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143001_test-whe505af1e71561618241641-2786283903-9c5cl-CLT-AL01_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "200919-154459_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");

    //testBenches.push_back(southwallBench2);

    RelocalizationTestBench augstBench = createRelocalizationTestBench("augst",
                                                                       "sunny",
                                                                       "map_augst_021019-115200_android-mcrd1-35-ASUS-A002.json",
                                                                       "ORBvoc.bin");

    addRelocalizationTestCase(augstBench, "021019-115146_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(augstBench, "021019-115233_android-mcrd1-35-ASUS-A002_640.mp4");

    testBenches.push_back(augstBench);

    RelocalizationTestBench southwallMarkerMapBench = createRelocalizationTestBench("southwall",
                                                                                    "shade",
                                                                                    "marker_map.json",
                                                                                    "ORBvoc.bin");

    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143003_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143004_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143001_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143002_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143001_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143002_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");

    testBenches.push_back(southwallMarkerMapBench);
    */

    for (int benchIndex = 0; benchIndex < testBenches.size(); benchIndex++)
    {
        RelocalizationTestBench b = testBenches[benchIndex];
        runRelocalizationTestBench(b);
    }

    return 0;
}
