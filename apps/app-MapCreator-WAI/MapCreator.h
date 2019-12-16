#ifndef MAP_CREATOR_H
#define MAP_CREATOR_H

#include <CVCapture.h>
#include <WAIHelper.h>
#include <Utils.h>
#include <AppWAISlamParamHelper.h>
#include <WAIModeOrbSlam2.h>
#include <WAIMapStorage.h>

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
    MapCreator(std::string erlebARDir, std::string configFile);
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
    void cullKeyframes(std::vector<WAIKeyFrame*>& kfs, const float cullRedundantPerc);
    void decorateDebug(WAI::ModeOrbSlam2* waiMode, CVCapture* cap, const int currentFrameIndex, const int videoLength, const int numOfKfs);
    void saveMap(WAI::ModeOrbSlam2* waiMode, const std::string& mapDir, const std::string& currentMapFileName, SLNode* mapNode = nullptr);
    void loadMap(WAI::ModeOrbSlam2* waiMode, const std::string& mapDir, const std::string& currentMapFileName, bool fixKfsForLBA, SLNode* mapNode);

    WAIFrame createMarkerFrame(std::string  markerFile,
                               KPextractor* markerExtractor);
    bool     findMarkerHomography(WAIFrame&    markerFrame,
                                  WAIKeyFrame* kfCand,
                                  cv::Mat&     homography,
                                  int          minMatches);

private:
    MapCreator() {}
    std::map<Location, Areas> _erlebAR;
    std::string               _erlebARDir;
    std::string               _vocFile;
    std::string               _calibrationsDir;
    std::string               _outputDir;

    bool doMarkerMapPreprocessing(const std::string& mapDir,
                                  const std::string& inputMapFile,
                                  std::string        markerFile,
                                  float              markerWidthInM,
                                  CVCalibration&     calib,
                                  const float        cullRedundantPerc);

    WAIMapPoint* _mpUL;
    WAIMapPoint* _mpUR;
    WAIMapPoint* _mpLL;
    WAIMapPoint* _mpLR;
};

#endif //MAP_CREATOR_H
