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



Tester::RelocalizationTestResult Tester::runRelocalizationTest(std::string videoFile,
                                                               std::string mapFile,
                                                               std::string vocFile,
                                                               CVCalibration &calibration)
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

    CVCapture::instance()->videoType(VT_FILE);
    CVCapture::instance()->videoFilename = videoFile;
    CVCapture::instance()->videoLoops    = false;

    CVSize2i videoSize       = CVCapture::instance()->openFile();
    float    widthOverHeight = (float)videoSize.width / (float)videoSize.height;
    std::unique_ptr<KPextractor> extractor  = _factory.make(7, {videoSize.width, videoSize.height});

    unsigned int lastRelocFrameId         = 0;
    int          frameCount               = 0;
    int          relocalizationFrameCount = 0;
    while (CVCapture::instance()->grabAndAdjustForSL(widthOverHeight))
    {
        cv::Mat intrinsic = calibration.cameraMat();
        cv::Mat distortion = calibration.distortion();
        WAIFrame currentFrame = WAIFrame(CVCapture::instance()->lastFrameGray,
                                         0.0f,
                                         extractor.get(),
                                         intrinsic, 
                                         distortion,
                                         orbVoc,
                                         false);

        int inliers;
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


void printHelp()
{
    std::stringstream ss;
    ss << "app-Test-WAI for creation of Erleb-AR maps!" << std::endl;
    ss << "Example1 (win):  app-Test-WAI.exe -erlebARDir C:/Erleb-AR -configFile testConfig.json" << std::endl;
    ss << "Example2 (unix): ./app-Test-WAI -erlebARDir ~/Erleb-AR -configFile testConfig.json" << std::endl;
    ss << "" << std::endl;
    ss << "Options: " << std::endl;
    ss << "  -h/-help        print this help, e.g. -h" << std::endl;
    ss << "  -erlebARDir     Path to Erleb-AR root directory" << std::endl;
    ss << "  -configFile     Path and name to TestConfig.json" << std::endl;
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
                Area area = (*itAreas)["area"];
                bool        enabled  = false;
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
                    std::string   name = *itVideos;
                    TestData testData;
                    testData.videoFile = erlebARDirUnified + "locations/" + location + "/" + area + "/" + "videos/" + name;
                    testData.mapFile = mapFile;

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
            RelocalizationTestResult r = runRelocalizationTest(testData.videoFile, testData.mapFile, _vocFile, testData.calibration);

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

        //init map creator
        DUtils::Random::SeedRandOnce(1337);
        Tester tester(config.erlebARDir, config.configFile, config.vocFile);
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
