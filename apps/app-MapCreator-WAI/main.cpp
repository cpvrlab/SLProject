#include <string>
#include <iostream>
#include "MapCreator.h"
#include <Utils.h>
#include <GLFW/glfw3.h>

//app parameter
struct Config
{
    std::string   erlebARDir;
    std::string   configFile;
    std::string   vocFile;
    std::string   mapOutputDir;
    ExtractorType extractorType;
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
    ss << "  -erlebARDir     Path to Erleb-AR root directory" << std::endl;
    ss << "  -configFile     Path and name to MapCreatorConfig.json" << std::endl;
    ss << "  -vocFile        Path and name to Vocabulary file" << std::endl;
    ss << "  -mapOutputDir   Directory where to output generated maps" << std::endl;

    std::cout << ss.str() << std::endl;
}

void readArgs(int argc, char* argv[], Config& config)
{
    config.extractorType = ExtractorType_GLSL;
    config.erlebARDir = Utils::getAppsWritableDir() + "/erleb-AR/";
    config.vocFile = config.erlebARDir + "voc/ORBvoc.bin";

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
        else if (!strcmp(argv[i], "-mapOutputDir"))
        {
            config.mapOutputDir = argv[++i]; //Not used
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
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-help"))
        {
            printHelp();
        }
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
        MapCreator mapCreator(config.erlebARDir, config.configFile, config.vocFile, config.extractorType);
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
