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

    WAIOrbVocabulary::initialize(vocFile);
    ORB_SLAM2::ORBVocabulary* orbVoc     = WAIOrbVocabulary::get();
    ORB_SLAM2::KPextractor*   extractor  = new ORB_SLAM2::SURFextractor(1500);
    WAIKeyFrameDB*            keyFrameDB = new WAIKeyFrameDB(*orbVoc);

    WAIMap* map = new WAIMap("map");
    WAIMapStorage::loadMap(map, keyFrameDB, nullptr, mapFile);

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

        if (WAI::ModeOrbSlam2::relocalization(currentFrame, keyFrameDB, &lastRelocFrameId))
        {
            relocalizationFrameCount++;
        }

        frameCount++;
    }

    result.frameCount               = frameCount;
    result.relocalizationFrameCount = relocalizationFrameCount;
    result.ratio                    = ((float)relocalizationFrameCount / (float)frameCount);
    result.wasSuccessful            = true; // TODO(dgj1): maybe check that files exist etc.

    return result;
}

void runRelocalizationTestBench(RelocalizationTestBench& testBench)
{
    testBench.testResults.resize(testBench.testCases.size());

    for (int i = 0; i < testBench.testCases.size(); i++)
    {
        RelocalizationTestCase testCase = testBench.testCases[i];
        testBench.testResults[i]        = runRelocalizationTest(
          Utils::getAppsWritableDir() + std::string("videos/") + testCase.videoFileName,
          Utils::getAppsWritableDir() + std::string("calibrations/") + testCase.calibrationFileName,
          Utils::getAppsWritableDir() + std::string("maps/") + testBench.mapFileName,
          std::string(SL_PROJECT_ROOT) + "/data/calibrations/" + testBench.vocFileName);
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

    RelocalizationTestBench southwallBench = createRelocalizationTestBench("southwall",
                                                                           "shade",
                                                                           "160919-143001_android-mcrd1-35-ASUS-A002.json",
                                                                           "ORBvoc.bin");

    addRelocalizationTestCase(southwallBench, "160919-143001_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143001_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143002_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143003_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143004_cm-cm-build-c25-TA-1021_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143001_test-whe505af1e71561618241641-2786283903-9c5cl-CLT-AL01_640.mp4");
    addRelocalizationTestCase(southwallBench, "160919-143001_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143002_android-mcrd1-35-ASUS-A002_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143001_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143002_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143003_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143004_cm-cm-build-c25-TA-1021_640.mp4", "camCalib_generic_smartphone.xml");
    addRelocalizationTestCase(southwallBench, "160919-143001_test-whe505af1e71561618241641-2786283903-9c5cl-CLT-AL01_640.mp4", "camCalib_generic_smartphone.xml");

    runRelocalizationTestBench(southwallBench);

    RelocalizationTestBench b = southwallBench;

    // print CSV
    for (int i = 0; i < southwallBench.testResults.size(); i++)
    {
        RelocalizationTestCase   c = southwallBench.testCases[i];
        RelocalizationTestResult r = southwallBench.testResults[i];

        if (r.wasSuccessful)
        {
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

    return 0;
}
