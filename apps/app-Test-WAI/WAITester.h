
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
        std::string map;
        std::string videoFile;
        CVCalibration calibration = {CVCameraType::VIDEOFILE, ""};
    } TestData;
    typedef std::vector<TestData> Datas;
    typedef std::map<Area, Datas> Areas;

public:
    Tester(std::string erlebARDir, std::string configFile, std::string vocFile);
    ~Tester();

    void launchTrackingTest(const Location& location, const Area& area, Datas& datas);

    void launchRelocalizationTest(const Location& location, const Area& area, Datas& datas);

    void execute();

    void loadSites(const std::string& erlebARDir, const std::string& configFile);

    void saveMap(WAISlam* waiMode, const std::string& mapDir, const std::string& currentMapFileName, SLNode* mapNode = nullptr);
    void loadMap(WAISlam* waiMode, const std::string& mapDir, const std::string& currentMapFileName, bool fixKfsForLBA, SLNode* mapNode);

private:
    Tester() {}
    std::map<Location, Areas> _erlebAR;
    std::string               _erlebARDir;
    std::string               _vocFile;
    std::string               _calibrationsDir;

    std::unique_ptr<KPextractor> _kpIniExtractor    = nullptr;
    std::unique_ptr<KPextractor> _kpExtractor       = nullptr;
};

#endif
