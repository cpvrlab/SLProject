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
        //std::string calibFile;
        CVCalibration calibration;
    };
    typedef std::vector<VideoAndCalib> Videos;
    typedef std::map<Area, Videos>     Areas;

public:
    MapCreator(std::string erlebARDir, std::string configFile)
    {
        //scan erlebar directory and config file, collect everything that is enabled in the config file and
        //check that all files (video and calibration) exist.
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
                throw std::runtime_error("Could not open configFile: " + configFile);

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
                            throw std::runtime_error("Video file does not exist: " + videoAndCalib.videoFile);

                        //check if calibration file exists
                        SlamVideoInfos slamVideoInfos;

                        if (!extractSlamVideoInfosFromFileName(name, &slamVideoInfos))
                            throw std::runtime_error("Could not extract slam video infos: " + name);

                        // construct calibrations file name and check if it exists
                        std::string calibDir  = erlebARDirUnified + "../calibrations/";
                        std::string calibFile = "camCalib_" + slamVideoInfos.deviceString + "_main.xml";

                        //videoAndCalib.calibFile = erlebARDirUnified + "../calibrations/" + "camCalib_" + slamVideoInfos.deviceString + "_main.xml";
                        if (!Utils::fileExists(calibDir + calibFile))
                            throw std::runtime_error("Calibration file does not exist: " + calibDir + calibFile);

                        //load calibration file and check for aspect ratio
                        if (!videoAndCalib.calibration.load(calibDir, calibFile, false, false))
                            throw std::runtime_error("Could not load calibration file: " + calibDir + calibFile);

                        std::vector<std::string> size;
                        Utils::splitString(slamVideoInfos.resolution, 'x', size);
                        if (size.size() == 2)
                        {
                            int width  = std::stoi(size[0]);
                            int height = std::stoi(size[1]);
                            if (videoAndCalib.calibration.imageSize().width != width ||
                                videoAndCalib.calibration.imageSize().height != height)
                            {
                                throw std::runtime_error("Resolutions of video and calibration do not fit together. Using: " + calibFile + " and " + name);
                            }
                        }
                        else
                        {
                            throw std::runtime_error("Could not estimate resolution string: " + calibFile);
                        }

                        //add video to videos vector
                        _erlebAR[location][area].push_back(videoAndCalib);
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

    void createWaiMap(const Area& area, Videos& videos)
    {
        CVCapture* cap = CVCapture::instance();
        cap->videoType(CVVideoType::VT_FILE);

        WAI_INFO("Starting map creation for area: %s", area.c_str());
        for (auto itVideos = videos.begin(); itVideos != videos.end(); ++itVideos)
        {
            cap->videoFilename    = itVideos->videoFile;
            cap->activeCalib      = &itVideos->calibration;
            cap->videoLoops       = false;
            cv::Size capturedSize = cap->openFile();

            if (capturedSize.width != cap->activeCalib->imageSize().width ||
                capturedSize.height != cap->activeCalib->imageSize().height)
                throw std::runtime_error("MapCreator::createWaiMap: Resolution of captured frame does not fit to calibration: " + itVideos->videoFile);
            float aspectRatio = cap->activeCalib->imageAspectRatio();

            while (cap->isOpened())
            {
                if (!cap->grabAndAdjustForSL(aspectRatio))
                    break;

#ifdef _DEBUG
                if (!cap->lastFrame.empty())
                {
                    cv::imshow("lastFrame", cap->lastFrame);
                    cv::waitKey(1);
                }
#endif
            }
        }
        WAI_INFO("Finished map creation for area: %s", area.c_str());
    }

    void execute()
    {
        try
        {
            for (auto itLocations = _erlebAR.begin(); itLocations != _erlebAR.end(); ++itLocations)
            {
                Areas& areas = itLocations->second;
                for (auto itAreas = areas.begin(); itAreas != areas.end(); ++itAreas)
                {
                    createWaiMap(itAreas->first, itAreas->second);
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
        mapCreator.execute();
    }
    catch (std::exception& e)
    {
        WAI_ERROR("Exception catched during map creation: %s", e.what());
    }
    catch (...)
    {
        WAI_ERROR("Unknown exception during map creation!");
    }

    return 0;
}
