#include <iostream>

#include <Utils.h>
#include <CVCapture.h>

#include <WAICalibration.h>
#include <WAIMapStorage.h>

#include <WAIFrame.h>
#include <WAIOrbVocabulary.h>
#include <WAIModeOrbSlam2.h>
#include <WAIKeyFrameDB.h>

#include <OrbSlam/KPextractor.h>
#include <OrbSlam/SURFextractor.h>

struct RelocalizationTestResult
{
    bool  wasSuccessful;
    int   frameCount;
    int   relocalizationFrameCount;
    float ratio;
};

struct RelocalizationTestCase
{
    std::string videoFileName;
    std::string calibrationFileName;
};

struct RelocalizationTestBench
{
    std::string location;
    std::string weatherConditions;

    std::string mapFileName;
    std::string vocFileName;

    std::vector<RelocalizationTestCase>   testCases;
    std::vector<RelocalizationTestResult> testResults;
};

bool getCalibrationFileNameFromVideoFileName(std::string  videoFileName,
                                             std::string* calibrationFileName)
{
    bool result = false;

    std::vector<std::string> stringParts;
    Utils::splitString(Utils::getFileNameWOExt(videoFileName), '_', stringParts);

    WAICalibration wc;
    if (stringParts.size() >= 3)
    {
        std::string computerInfo = stringParts[1];
        *calibrationFileName     = "camCalib_" + computerInfo + "_main.xml";
        result                   = true;
    }

    return result;
}

RelocalizationTestResult runRelocalizationTest(std::string videoFile,
                                               std::string calibrationFile,
                                               std::string mapFile,
                                               std::string vocFile)
{
    RelocalizationTestResult result = {};

    //TODO FIX NOW
    /*
    // TODO(dgj1): this is kind of a hack... improve (maybe separate function call??)
    WAIFrame::mbInitialComputations = true;

    WAIOrbVocabulary::initialize(vocFile);
    ORB_SLAM2::ORBVocabulary* orbVoc     = WAIOrbVocabulary::get();
    ORB_SLAM2::KPextractor*   extractor  = new ORB_SLAM2::SURFextractor(800);
    WAIKeyFrameDB*            keyFrameDB = new WAIKeyFrameDB(*orbVoc);

    WAIMap* map = new WAIMap(keyFrameDB);
    WAIMapStorage::loadMap(map, nullptr, mapFile, false, true);

    CVCapture::instance()->videoType(VT_FILE);
    CVCapture::instance()->videoFilename = videoFile;
    CVCapture::instance()->videoLoops    = false;

    CVSize2i videoSize       = CVCapture::instance()->openFile();
    float    widthOverHeight = (float)videoSize.width / (float)videoSize.height;

    WAICalibration wc;
    wc.loadFromFile(calibrationFile);

    unsigned int lastRelocFrameId         = 0;
    int          frameCount               = 0;
    int          relocalizationFrameCount = 0;
    while (CVCapture::instance()->grabAndAdjustForSL(widthOverHeight))
    {
        WAIFrame currentFrame = WAIFrame(CVCapture::instance()->lastFrameGray,
                                         0.0f,
                                         extractor,
                                         wc.cameraMat(),
                                         wc.distortion(),
                                         orbVoc,
                                         false);

        if (WAISlam::relocalization(currentFrame, keyFrameDB, &lastRelocFrameId, *map, false))
        {
            relocalizationFrameCount++;
        }

        frameCount++;
    }

    result.frameCount               = frameCount;
    result.relocalizationFrameCount = relocalizationFrameCount;
    result.ratio                    = ((float)relocalizationFrameCount / (float)frameCount);
    result.wasSuccessful            = true;
    */

    return result;
}

void runRelocalizationTestBench(RelocalizationTestBench& b)
{
    b.testResults.resize(b.testCases.size());

    for (int i = 0; i < b.testCases.size(); i++)
    {
        RelocalizationTestCase   c = b.testCases[i];
        RelocalizationTestResult r = runRelocalizationTest(
          Utils::getAppsWritableDir() + std::string("videos/") + c.videoFileName,
          Utils::getAppsWritableDir() + std::string("calibrations/") + c.calibrationFileName,
          Utils::getAppsWritableDir() + std::string("maps/") + b.mapFileName,
          Utils::getAppsWritableDir() + std::string("voc/") + b.vocFileName);

        // scene;conditions;videoFileName;calibrationName;mapName;vocName;frameCount;relocalizationFrameCount;ratio
        printf("%s;%s;%s;%s;%s;%s;%i;%i;%.2f\n",
               b.location.c_str(),
               b.weatherConditions.c_str(),
               c.videoFileName.c_str(),
               c.calibrationFileName.c_str(),
               b.mapFileName.c_str(),
               b.vocFileName.c_str(),
               r.frameCount,
               r.relocalizationFrameCount,
               r.ratio);
    }
}

void addRelocalizationTestCase(std::vector<RelocalizationTestCase>& tests,
                               std::string                          videoFileName,
                               std::string                          calibrationFileName = "")
{
    RelocalizationTestCase test = {};

    test.videoFileName = videoFileName;

    if (calibrationFileName.empty())
    {
        if (getCalibrationFileNameFromVideoFileName(test.videoFileName, &test.calibrationFileName))
        {
            tests.push_back(test);
        }
    }
    else
    {
        test.calibrationFileName = calibrationFileName;
        tests.push_back(test);
    }
}

void addRelocalizationTestCase(RelocalizationTestBench& testBench,
                               std::string              videoFileName,
                               std::string              calibrationFileName = "")
{
    addRelocalizationTestCase(testBench.testCases, videoFileName, calibrationFileName);
}

RelocalizationTestBench createRelocalizationTestBench(std::string location,
                                                      std::string weatherConditions,
                                                      std::string mapFileName,
                                                      std::string vocFileName)
{
    RelocalizationTestBench result = {};

    result.location          = location;
    result.weatherConditions = weatherConditions;
    result.mapFileName       = mapFileName;
    result.vocFileName       = vocFileName;

    return result;
}

int main()
{
    DUtils::Random::SeedRandOnce(1337);

    std::vector<RelocalizationTestBench> testBenches;

    RelocalizationTestBench southwallBench = createRelocalizationTestBench("southwall",
                                                                           "shade",
                                                                           "160919-143001_android-mcrd1-35-ASUS-A002.json",
                                                                           "ORBvoc.bin");

    addRelocalizationTestCase(southwallBench, "160919-143001_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143003_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143004_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143001_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143002_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143003_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143004_cm-cm-build-c25-TA-1021_640.mp4");
    //addRelocalizationTestCase(southwallBench, "160919-143001_test-whe505af1e71561618241641-2786283903-9c5cl-CLT-AL01_640.mp4");
    addRelocalizationTestCase(southwallBench, "200919-154459_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench, "011019-164748_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench, "011019-164844_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench, "011019-165120_cm-cm-build-c25-TA-1021_640.mp4");

    /*
    addRelocalizationTestCase(southwallBench, "160919-143001_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143001_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143002_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143003_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143004_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143001_test-whe505af1e71561618241641-2786283903-9c5cl-CLT-AL01_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "200919-154459_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
*/

    //testBenches.push_back(southwallBench);

    RelocalizationTestBench southwallBench2 = createRelocalizationTestBench("southwall",
                                                                            "shade",
                                                                            "160919-143001_android-mcrd1-35-ASUS-A002.json",
                                                                            "voc_southwall_200919_154459_android-mcrd1-35-ASUS-A002_640.bin");

    addRelocalizationTestCase(southwallBench2, "160919-143001_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143003_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143004_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143001_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143002_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143003_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallBench2, "160919-143004_cm-cm-build-c25-TA-1021_640.mp4");
    //addRelocalizationTestCase(southwallBench2, "160919-143001_test-whe505af1e71561618241641-2786283903-9c5cl-CLT-AL01_640.mp4");
    addRelocalizationTestCase(southwallBench2, "200919-154459_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "011019-164748_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "011019-164844_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench2, "011019-165120_cm-cm-build-c25-TA-1021_640.mp4");

    /*
    addRelocalizationTestCase(southwallBench2, "160919-143001_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143001_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143002_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143003_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143004_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "160919-143001_test-whe505af1e71561618241641-2786283903-9c5cl-CLT-AL01_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench2, "200919-154459_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    */

    //testBenches.push_back(southwallBench2);

    RelocalizationTestBench augstBench = createRelocalizationTestBench("augst",
                                                                       "sunny",
                                                                       "map_augst_021019-115200_android-mcrd1-35-ASUS-A002.json",
                                                                       "ORBvoc.bin");

    addRelocalizationTestCase(augstBench, "021019-115146_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(augstBench, "021019-115233_android-mcrd1-35-ASUS-A002_640.mp4");

    testBenches.push_back(augstBench);

    RelocalizationTestBench southwallMarkerMapBench = createRelocalizationTestBench("southwall",
                                                                                    "shade",
                                                                                    "marker_map.json",
                                                                                    "ORBvoc.bin");

    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143003_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143004_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143001_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143002_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143001_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallMarkerMapBench, "160919-143002_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");

    testBenches.push_back(southwallMarkerMapBench);

    for (int benchIndex = 0; benchIndex < testBenches.size(); benchIndex++)
    {
        RelocalizationTestBench b = testBenches[benchIndex];
        runRelocalizationTestBench(b);
    }

    return 0;
}
