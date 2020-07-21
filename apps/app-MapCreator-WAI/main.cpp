#include <string>
#include <iostream>
#include <MapCreator.h>
#include <Utils.h>
#include <GLFW/glfw3.h>

//app parameter
struct Config
{
    std::string   erlebARDir;
    std::string   calibrationsDir;
    std::string   configFile;
    std::string   vocFile;
    std::string   outputDir;
    ExtractorType extractorType;
    int           nLevels;
    bool          serialMapping;
    float         thinCullingValue;
};

void printHelp()
{
    std::stringstream ss;
    ss << "app-MapCreator for creation of Erleb-AR maps!" << std::endl;
    ss << "Example1 (win):  app-MapCreator.exe -erlebARDir C:/Erleb-AR -configFile MapCreatorConfig.json -mapOutputDir output" << std::endl;
    ss << "Example2 (unix): ./app-MapCreator -erlebARDir C:/Erleb-AR -configFile MapCreatorConfig.json -mapOutputDir output" << std::endl;
    ss << "" << std::endl;
    ss << "Options: " << std::endl;
    ss << "  -h/-help        print this help, e.g. -h" << std::endl;
    ss << "  -erlebARDir     Path to Erleb-AR root directory (Optional. If not specified, <AppsWritableDir>/erleb-AR/ is used)" << std::endl;
    ss << "  -calibDir       Path to directory containing camera calibrations (Optional. If not specified, <AppsWritableDir>/voc/voc_fbow.bin is used)" << std::endl;
    ss << "  -configFile     Path and name to MapCreatorConfig.json" << std::endl;
    ss << "  -vocFile        Path and name to Vocabulary file (Optional. If not specified, <AppsWritableDir>/calibrations/ is used)" << std::endl;
    ss << "  -outputDir      Directory where to output generated data (maps, log). (Optional. If not specified, <erlebARDir>/MapCreator/ is used for log output)" << std::endl;
    ss << "  -levels         Number of pyramid levels" << std::endl;
    ss << "  -serial         Serial mapping (1 or 0)" << std::endl;
    ss << "  -thinCullVal    Thin out culling value (e.g. 0.95)" << std::endl;

    std::cout << ss.str() << std::endl;
}

void readArgs(int argc, char* argv[], Config& config)
{
    config.extractorType    = ExtractorType_FAST_BRIEF_1000;
    config.erlebARDir       = Utils::getAppsWritableDir() + "erleb-AR/";
    config.calibrationsDir  = Utils::getAppsWritableDir() + "calibrations/";
    config.nLevels          = -1;
    config.thinCullingValue = 0.995f;
    config.serialMapping    = false;

#if USE_FBOW
    config.vocFile = Utils::getAppsWritableDir() + "voc/voc_fbow.bin";
#else
    config.vocFile = Utils::getAppsWritableDir() + "voc/ORBvoc.bin";
#endif

    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "-erlebARDir"))
        {
            config.erlebARDir = argv[++i];
        }
        else if (!strcmp(argv[i], "-calibDir"))
        {
            config.calibrationsDir = argv[++i];
        }
        else if (!strcmp(argv[i], "-configFile"))
        {
            config.configFile = argv[++i];
        }
        else if (!strcmp(argv[i], "-vocFile"))
        {
            config.vocFile = argv[++i];
        }
        else if (!strcmp(argv[i], "-outputDir"))
        {
            config.outputDir = argv[++i]; //Not used
        }
        else if (!strcmp(argv[i], "-level"))
        {
            config.nLevels = std::stoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-serial"))
        {
            int val              = std::stoi(argv[++i]);
            config.serialMapping = val == 1 ? true : false;
        }
        else if (!strcmp(argv[i], "-thinCullVal"))
        {
            config.thinCullingValue = std::stof(argv[++i]);
        }
        else if (!strcmp(argv[i], "-feature"))
        {
            i++;
            if (!strcmp(argv[i], "FAST_BRIEF_1000"))
                config.extractorType = ExtractorType_FAST_BRIEF_1000;
            else if (!strcmp(argv[i], "FAST_BRIEF_2000"))
                config.extractorType = ExtractorType_FAST_BRIEF_2000;
            else if (!strcmp(argv[i], "FAST_BRIEF_4000"))
                config.extractorType = ExtractorType_FAST_BRIEF_4000;
            else if (!strcmp(argv[i], "FAST_BRIEF_6000"))
                config.extractorType = ExtractorType_FAST_BRIEF_6000;
            else if (!strcmp(argv[i], "FAST_ORBS_1000"))
                config.extractorType = ExtractorType_FAST_ORBS_1000;
            else if (!strcmp(argv[i], "FAST_ORBS_2000"))
                config.extractorType = ExtractorType_FAST_ORBS_2000;
            else if (!strcmp(argv[i], "FAST_ORBS_4000"))
                config.extractorType = ExtractorType_FAST_ORBS_4000;
            else if (!strcmp(argv[i], "FAST_ORBS_6000"))
                config.extractorType = ExtractorType_FAST_ORBS_6000;
            else if (!strcmp(argv[i], "GLSL_1"))
            {
                config.extractorType = ExtractorType_GLSL_1;
                config.nLevels       = 1;
            }
            else if (!strcmp(argv[i], "GLSL"))
            {
                config.extractorType = ExtractorType_GLSL;
                config.nLevels       = 1;
            }
            else
            {
                std::cerr << "unkown feature type" << std::endl;
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-help"))
        {
            printHelp();
        }
    }
    if (config.nLevels == -1)
    {
        std::cerr << "pyramid level not specified" << std::endl;
        exit(1);
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

#ifdef __APPLE__
    //You can enable or restrict newer OpenGL context here (read the GLFW documentation)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

    window = glfwCreateWindow(scrWidth, scrHeight, "My Title", nullptr, nullptr);

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
    if (gl3wInit() != 0)
    {
        cerr << "Failed to initialize OpenGL" << endl;
        exit(-1);
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
        if (config.outputDir.empty())
        {
            Utils::initFileLog(Utils::unifySlashes(config.erlebARDir) + "MapCreator/", true);
        }
        else
        {
            Utils::initFileLog(Utils::unifySlashes(config.outputDir) + "log/", true);
        }
        Utils::log("Main", "MapCreator");

        //init map creator
        MapCreator mapCreator(config.erlebARDir,
                              config.calibrationsDir,
                              config.configFile,
                              config.vocFile,
                              config.extractorType,
                              config.nLevels,
                              config.outputDir,
                              config.serialMapping,
                              config.thinCullingValue);
        //todo: call different executes e.g. executeFullProcessing(), executeThinOut()
        //input and output directories have to be defined together with json file which is always scanned during construction
        mapCreator.execute();
    }
    catch (std::exception& e)
    {
        Utils::log("Main", "Exception catched during map creation: %s", e.what());
    }
    catch (...)
    {
        Utils::log("Main", "Unknown exception during map creation!");
    }

    return 0;
}
