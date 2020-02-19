#ifndef MAP_CREATOR_H
#define MAP_CREATOR_H

#include <CVCapture.h>
#include <WAIHelper.h>
#include <Utils.h>
#include <AppWAISlamParamHelper.h>
#include <WAISlam.h>
#include <WAIMapStorage.h>

#define WAI_DEBUG(...) Utils::log("[DEBUG]", __VA_ARGS__)
#define WAI_INFO(...) Utils::log("[INFO ]", __VA_ARGS__)
#define WAI_WARN(...) Utils::log("[WARN ]", __VA_ARGS__)

//#define WAI_DEBUG(...) // nothing
//#define WAI_INFO(...)  // nothing
//#define WAI_WARN(...)  // nothing

class MapCreator
{
    typedef std::string Location;
    typedef std::string Area;
    typedef struct VideoAndCalib
    {
        std::string videoFile;
        //std::string calibFile;
        CVCalibration calibration = {CVCameraType::VIDEOFILE, ""};
    };
    typedef std::vector<VideoAndCalib> Videos;
    typedef struct AreaConfig
    {
        Videos      videos;
        std::string markerFile;
    };
    typedef std::map<Area, AreaConfig> Areas;

public:
    MapCreator(std::string erlebARDir, std::string configFile, std::string vocFile);
    ~MapCreator();
    //! execute map creation
    void execute();
    //! Scan erlebar directory and config file, collect everything that is enabled in the config file and
    //! check that all files (video and calibration) exist.
    void loadSites(const std::string& erlebARDir, const std::string& configFile);
    //! create dense map using all videos for this location/area and thin out overall resulting map using keyframe culling
    void createNewWaiMap(const Location& location, const Area& area, AreaConfig& areaConfig);

    bool createNewDenseWaiMap(Videos&            videos,
                              const std::string& mapFile,
                              const std::string& mapDir,
                              const float        cullRedundantPerc,
                              std::string&       currentMapFileName);
    void thinOutNewWaiMap(const std::string& mapDir,
                          const std::string& inputMapFile,
                          const std::string  outputMapFile,
                          CVCalibration&     calib,
                          const float        cullRedundantPerc);
    void cullKeyframes(WAISlam* waiMode, std::vector<WAIKeyFrame*>& kfs, const float cullRedundantPerc);
    void decorateDebug(WAISlam* waiMode, CVCapture* cap, const int currentFrameIndex, const int videoLength, const int numOfKfs);
    void saveMap(WAISlam* waiMode, const std::string& mapDir, const std::string& currentMapFileName, SLNode* mapNode = nullptr);
    void loadMap(WAISlam* waiMode, const std::string& mapDir, const std::string& currentMapFileName, bool fixKfsForLBA, SLNode* mapNode);

    bool createMarkerMap(AreaConfig&        areaConfig,
                         const std::string& mapFile,
                         const std::string& mapDir,
                         const float        cullRedundantPerc);

private:
    MapCreator() {}
    std::map<Location, Areas> _erlebAR;
    std::string               _erlebARDir;
    std::string               _vocFile;
    std::string               _calibrationsDir;
    std::string               _outputDir;

    WAIMapPoint* _mpUL;
    WAIMapPoint* _mpUR;
    WAIMapPoint* _mpLL;
    WAIMapPoint* _mpLR;

    std::unique_ptr<KPextractor> _kpIniExtractor    = nullptr;
    std::unique_ptr<KPextractor> _kpExtractor       = nullptr;
    std::unique_ptr<KPextractor> _kpMarkerExtractor = nullptr;
};

#endif //MAP_CREATOR_H
