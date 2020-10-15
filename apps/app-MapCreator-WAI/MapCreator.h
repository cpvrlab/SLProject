#ifndef MAP_CREATOR_H
#define MAP_CREATOR_H

#include <CVCapture.h>
#include <WAIHelper.h>
#include <Utils.h>
#include <AppWAISlamParamHelper.h>
#include <WAIMapSlam.h>
#include <WAIMapStorage.h>
#include <FeatureExtractorFactory.h>
#include <WAIOrbVocabulary.h>

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
        std::string   videoFile;
        CVCalibration calibration = {CVCameraType::VIDEOFILE, ""};
    } VideoAndCalib;
    typedef std::vector<VideoAndCalib> Videos;
    typedef struct AreaConfig
    {
        Videos      videos;
        std::string markerFile;
        std::string initialMapFile;
    } AreaConfig;
    typedef std::map<Area, AreaConfig> Areas;

public:
    MapCreator(std::string   erlebARDir,
               std::string   calibrationsDir,
               std::string   configFile,
               std::string   vocFile,
               int           vocLevels,
               ExtractorType extractorType,
               int           nLevels,
               std::string   outputDir,
               bool          serialMapping,
               float         thinCullingValue,
               bool          ensureKFIntegration,
               float         minCommonWordFactor,
               bool          saveBinary);
    ~MapCreator();
    //! execute map creation
    void execute();
    //! Scan erlebar directory and config file, collect everything that is enabled in the config file and
    //! check that all files (video and calibration) exist.
    void loadSites(const std::string& erlebARDir, const std::string& configFile);
    //! create dense map using all videos for this location/area and thin out overall resulting map using keyframe culling
    void createNewWaiMap(const Location& location, const Area& area, AreaConfig& areaConfig, ExtractorType extractorType, int nLevels, bool ensureKFIntegration);

    bool createNewDenseWaiMap(Videos&                   videos,
                              const std::string&        mapFile,
                              const std::string&        mapDir,
                              const float               cullRedundantPerc,
                              std::string&              currentMapFileName,
                              ExtractorType             extractorType,
                              int                       nLevels,
                              std::vector<int>&         keyFrameVideoMatching,
                              std::vector<std::string>& matchFileVideoNames, //video names loaded from match file
                              const std::string&        initialMapFileName,
                              bool                      ensureKFIntegration);

    void thinOutNewWaiMap(const std::string&              mapDir,
                          const std::string&              inputMapFile,
                          const std::string&              outputMapFile,
                          const std::string&              outputKFMatchingFile,
                          CVCalibration&                  calib,
                          const float                     cullRedundantPerc,
                          ExtractorType                   extractorType,
                          int                             nLevels,
                          std::vector<int>&               keyFrameVideoMatching,
                          const std::vector<std::string>& allVideos);

    bool createMarkerMap(AreaConfig&        areaConfig,
                         const std::string& mapFile,
                         const std::string& mapDir,
                         const float        cullRedundantPerc,
                         ExtractorType      extractorType,
                         int                nLevels);

    void cullKeyframes(WAIMap* map, std::vector<WAIKeyFrame*>& kfs, std::vector<int>& keyFrameVideoMatching, const float cullRedundantPerc);
    void decorateDebug(WAIMapSlam* waiMode, cv::Mat lastFrame, const int currentFrameIndex, const int videoLength, const int numOfKfs);
    void saveMap(WAIMapSlam* waiMode, const std::string& mapDir, const std::string& currentMapFileName, SLNode* mapNode = nullptr);

private:
    MapCreator() {}
    std::map<Location, Areas> _erlebAR;
    std::string               _erlebARDir;
    std::string               _calibrationsDir;
    std::string               _outputDir;
    WAIOrbVocabulary*         _voc = nullptr;
    bool                      _ensureKFIntegration;

    WAIMapPoint* _mpUL = nullptr;
    WAIMapPoint* _mpUR = nullptr;
    WAIMapPoint* _mpLL = nullptr;
    WAIMapPoint* _mpLR = nullptr;

    ExtractorType _extractorType = ExtractorType_FAST_ORBS_2000;
    int           _nLevels       = 2;

    bool  _serialMapping       = false;
    float _thinCullingValue    = 0.995f;
    bool  _saveBinary       = false;
    float _minCommonWordFactor = 0.8;
    /*
    std::unique_ptr<KPextractor> _kpIniExtractor    = nullptr;
    std::unique_ptr<KPextractor> _kpExtractor       = nullptr;
    std::unique_ptr<KPextractor> _kpMarkerExtractor = nullptr;
    */
};

#endif //MAP_CREATOR_H
