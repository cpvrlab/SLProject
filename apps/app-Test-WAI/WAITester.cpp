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

void runRelocalizationTest(std::string videoFileName)
{
    WAIOrbVocabulary::initialize(std::string(SL_PROJECT_ROOT) + "/data/calibrations/ORBvoc.bin"); // TODO(dgj1): possibility to change voc
    ORB_SLAM2::ORBVocabulary* orbVoc     = WAIOrbVocabulary::get();
    ORB_SLAM2::KPextractor*   extractor  = new ORB_SLAM2::SURFextractor(1500);
    WAIKeyFrameDB*            keyFrameDB = new WAIKeyFrameDB(*orbVoc);

    WAIMap* map = new WAIMap("map");
    WAIMapStorage::loadMap(map, keyFrameDB, nullptr, Utils::getAppsWritableDir() + std::string("maps/160919-143001_android-mcrd1-35-ASUS-A002.json"));

    CVCapture::instance()->videoType(VT_FILE);
    CVCapture::instance()->videoFilename = Utils::getAppsWritableDir() + std::string("/videos/") + videoFileName;
    CVCapture::instance()->videoLoops    = false;
    CVCapture::instance()->openFile();

    // get calibration file name from video file name
    std::vector<std::string> stringParts;
    Utils::splitString(Utils::getFileNameWOExt(videoFileName), '_', stringParts);

    WAICalibration wc;
    if (stringParts.size() >= 3)
    {
        std::string computerInfo = stringParts[1];
        wc.loadFromFile(Utils::getAppsWritableDir() + std::string("/calibrations/") + "camCalib_" + computerInfo + "_main.xml");
    }

    unsigned int lastRelocFrameId         = 0;
    int          frameCount               = 0;
    int          relocalizationFrameCount = 0;
    while (CVCapture::instance()->grabAndAdjustForSL(640.0f / 360.0f)) // TODO(dgj1): actual size from video
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

    printf("Could relocalize in %i of %i frames (%.0f%%)\n",
           relocalizationFrameCount,
           frameCount,
           ((float)relocalizationFrameCount / (float)frameCount) * 100.0f);
}

int main()
{
    runRelocalizationTest("160919-143001_android-mcrd1-35-ASUS-A002_640.mp4");
    runRelocalizationTest("160919-143002_android-mcrd1-35-ASUS-A002_640.mp4");
    runRelocalizationTest("160919-143003_android-mcrd1-35-ASUS-A002_640.mp4");
    runRelocalizationTest("160919-143004_android-mcrd1-35-ASUS-A002_640.mp4");

    return 0;
}