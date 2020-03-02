
#ifndef TESTER_H
#define TESTER_H

#include <CVCapture.h>
#include <Utils.h>
#include <WAISlam.h>
#include <WAIMapStorage.h>

#define WAI_DEBUG(...) Utils::log("[DEBUG]", __VA_ARGS__)
#define WAI_INFO(...) Utils::log("[INFO ]", __VA_ARGS__)
#define WAI_WARN(...) Utils::log("[WARN ]", __VA_ARGS__)

class Tester
{
    typedef std::string Location;
    typedef std::string Area;
    typedef struct TestData
    {
        std::string   mapFile;
        std::string   videoFile;
        CVCalibration calibration = {CVCameraType::VIDEOFILE, ""};
    } TestData;
    typedef std::vector<TestData> Datas;
    typedef std::map<Area, Datas> Areas;

    struct RelocalizationTestResult
    {
        bool  wasSuccessful;
        int   frameCount;
        int   relocalizationFrameCount;
        float ratio;
    };

    struct TrackingTestResult
    {
        bool  wasSuccessful;
        int   frameCount;
        int   trackingFrameCount;
        float ratio;
    };

public:
    Tester(std::string erlebARDir, std::string configFile, std::string vocFile, int testFlags, int frameRate, ExtractorType extractorType);
    ~Tester();

    RelocalizationTestResult runRelocalizationTest(std::string    videoFile,
                                                   std::string    mapFile,
                                                   std::string    vocFile,
                                                   CVCalibration& calibration,
                                                   ExtractorType  extractorType);

    TrackingTestResult runTrackingTest(std::string    videoFile,
                                       std::string    mapFile,
                                       std::string    vocFile,
                                       CVCalibration& calibration,
                                       ExtractorType  extractorType,
                                       int            framerate = 0);

    void launchTrackingTest(const Location& location, const Area& area, Datas& datas, ExtractorType extractorType, int framerate = 0);

    void launchRelocalizationTest(const Location& location, const Area& area, Datas& datas, ExtractorType extractorType);

    void execute();

    void loadSites(const std::string& erlebARDir, const std::string& configFile);

    void saveMap(WAISlam* waiMode, const std::string& mapDir, const std::string& currentMapFileName, SLNode* mapNode = nullptr);
    void loadMap(WAISlam* waiMode, const std::string& mapDir, const std::string& currentMapFileName, bool fixKfsForLBA, SLNode* mapNode);

private:
    Tester() {}
    std::map<Location, Areas> _erlebAR;
    std::string               _erlebARDir;
    std::string               _vocFile;
    int                       _testFlags;
    std::string               _calibrationsDir;
    FeatureExtractorFactory   _factory;
    int                       _framerate; // (#frames/s)
    ExtractorType             _extractorType;
};

#endif
