#include <string>
#include <iostream>
#include "MapCreator.h"

//app parameter
struct Config
{
    std::string erlebARDir;
    std::string configFile;
    std::string mapOutputDir;
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
    ss << "  -mapOutputDir   Directory where to output generated maps" << std::endl;

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
        else if (!strcmp(argv[i], "-mapOutputDir"))
        {
            config.mapOutputDir = argv[++i];
        }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-help"))
        {
            printHelp();
        }
    }
}

int main(int argc, char* argv[])
{
    try
    {
        Config config;

        //parse arguments
        readArgs(argc, argv, config);

        //initialize logger
        std::string cwd = Utils::getCurrentWorkingDir();
        Logger::initFileLog(Utils::unifySlashes(config.erlebARDir) + "MapCreator/", true);
        WAI_INFO("WAI MapCreator");

        //init map creator
        MapCreator mapCreator(config.erlebARDir, config.configFile);
        //todo: call different executes e.g. executeFullProcessing(), executeThinOut()
        //input and output directories have to be defined together with json file which is always scanned during construction
        mapCreator.execute();
    }
    catch (std::exception& e)
    {
        WAI_ERROR("Exception catched during map creation: %s", e.what());
        Logger::flushFileLog();
    }
    catch (...)
    {
        WAI_ERROR("Unknown exception during map creation!");
        Logger::flushFileLog();
    }

    return 0;
}
