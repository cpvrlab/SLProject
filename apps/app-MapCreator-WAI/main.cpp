#include <CVCapture.h>
#include <WAIHelper.h>
#include <Utils.h>
#include <AppWaiSlamParamHelper.h>

class MapCreator
{
    typedef std::string Location;
    typedef std::string Area;
    typedef struct VideoAndCalib
    {
        std::string videoFile;
        std::string calibFile;
    };
    typedef std::vector<VideoAndCalib> Videos;
    typedef std::map<Area, Videos>     Areas;

public:
    MapCreator(std::string erlebARDir, std::string configFile)
    {
        loadSites(erlebARDir, configFile);
    }

    void loadSites(const std::string& erlebARDir, const std::string& configFile)
    {
        try
        {
            WAI_DEBUG("MapCreator: loading sites:");
            //parse config file
            cv::FileStorage fs;
            fs.open(configFile, cv::FileStorage::READ);
            if (!fs.isOpened())
                throw std::runtime_error("MapCreator: loadSites: Could not open configFile: " + configFile);

            //helper for areas that have been enabled
            std::set<Area> enabledAreas;

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
                        WAI_DEBUG("enabling %s %s", location.c_str(), area.c_str());
                        Areas& areas = _erlebAR[location];
                        //insert empty Videos vector
                        areas.insert(std::pair<std::string, std::vector<VideoAndCalib>>(area, Videos()));
                        enabledAreas.insert(area);
                    }
                }
            }

            std::string erlebARDirUnified = Utils::unifySlashes(erlebARDir);
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
                        VideoAndCalib videoAndCalib;
                        videoAndCalib.videoFile = erlebARDirUnified + "locations/" + location + "/" + area + "/" + "videos/" + name;

                        if (!Utils::fileExists(videoAndCalib.videoFile))
                            throw std::runtime_error("MapCreator::loadSites: Video file does not exist: " + videoAndCalib.videoFile);

                        //check if calibration file exists
                        SlamVideoInfos slamVideoInfos;

                        if (!extractSlamVideoInfosFromFileName(name, &slamVideoInfos))
                            throw std::runtime_error("MapCreator::loadSites: Could not extract slam video infos: " + name);

                        // construct calibrations file name and check if it exists
                        videoAndCalib.calibFile = erlebARDirUnified + "../calibrations/" + "camCalib_" + slamVideoInfos.deviceString + "_main.xml";
                        if (!Utils::fileExists(videoAndCalib.calibFile))
                            throw std::runtime_error("MapCreator::loadSites: Calibration file does not exist: " + videoAndCalib.calibFile);

                        //add video to videos vector
                        _erlebAR[location][area].push_back(videoAndCalib);
                    }
                }
            }

            //Check that there is a corresponding calibration file for every video:
            //We extract the device name and the resolution from the video file name and search for a calibration file from this device with the same resolution.

            //prepare content of calibrations directory:
            //(maps by device a vector of resolutions and the fullpath of the calibrations file)
            //struct Calibrations
            //{
            //    bool operator==(const Calibrations& other)
            //    {
            //        return (resolution.width == other.resolution.width &&
            //                resolution.height == other.resolution.height);
            //    }
            //    cv::Size    resolution;
            //    std::string fileName;
            //    std::string deviceName;
            //};
            //std::map<std::string, std::vector<Calibrations>> calibrations;
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
