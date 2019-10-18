#include <CVCapture.h>
#include <WAIHelper.h>
#include <Utils.h>

class MapCreator
{
    typedef std::string              Location;
    typedef std::string              Area;
    typedef std::vector<std::string> Videos;
    typedef std::map<Area, Videos>   Areas;

public:
    MapCreator(std::string erlebARDir, std::string configFile)
    {
        loadSites(erlebARDir, configFile);
    }

    void loadSites(const std::string& erlebARDir, const std::string& configFile)
    {
        try
        {
            //parse config file
            cv::FileStorage fs;
            fs.open(configFile, cv::FileStorage::READ);
            if (!fs.isOpened())
                throw std::runtime_error("MapCreator: loadSites: Could not open configFile: " + configFile);

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
                        Areas& areas = _erlebAR[location];
                        //insert empty Videos vector
                        areas.insert(std::pair<std::string, std::vector<std::string>>(area, Videos()));
                    }
                }
            }

            //try to find corresponding files in sites directory and add full file paths to _sites
            for (auto itLocs = _erlebAR.begin(); itLocs != _erlebAR.end(); ++itLocs)
            {
                Location location = itLocs->first;
                Areas&   areas    = itLocs->second;
                for (auto itAreas = areas.begin(); itAreas != areas.end(); ++itAreas)
                {
                }
            }

            //check that there is a corresponding calibration file
        }
        catch (std::exception& e)
        {
            throw std::runtime_error("Exception in MapCreator::loadSites: " + std::string(e.what()));
        }
        catch (...)
        {
            throw std::runtime_error("Unknown exception catched in MapCreator!");
        }
    }

private:
    MapCreator() {}
    std::map<Location, Areas> _erlebAR;
};

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
        Logger::initFileLog(cwd, false);
        WAI_INFO("WAI MapCreator");

        //init map creator
        MapCreator mapCreator(config.erlebARDir, config.configFile);

        //CVCapture* cap = CVCapture::instance();
        //cap->activeCalib
    }
    catch (std::exception& e)
    {
        WAI_ERROR("Exception catched during map creation: ", e.what());
    }
    catch (...)
    {
        WAI_ERROR("Unknown exception during map creation!");
    }

    return 0;
}
