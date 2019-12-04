#include <atomic>

#include <SLApplication.h>
#include <SLInterface.h>
#include <SLKeyframeCamera.h>
#include <CVCapture.h>
#include <Utils.h>
#include <AverageTiming.h>

#include <WAIModeOrbSlam2.h>
#include <WAIMapStorage.h>
#include <WAICalibration.h>
#include <AppWAIScene.h>
#include <AppDemoGui.h>
#include <AppDemoGuiMenu.h>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiAbout.h>
#include <AppDemoGuiInfosFrameworks.h>
#include <AppDemoGuiInfosMapNodeTransform.h>
#include <AppDemoGuiInfosScene.h>
#include <AppDemoGuiInfosSensors.h>
#include <AppDemoGuiInfosTracking.h>
#include <AppDemoGuiProperties.h>
#include <AppDemoGuiSceneGraph.h>
#include <AppDemoGuiStatsDebugTiming.h>
#include <AppDemoGuiStatsTiming.h>
#include <AppDemoGuiStatsVideo.h>
#include <AppDemoGuiTrackedMapping.h>
#include <AppDemoGuiTransform.h>
#include <AppDemoGuiUIPrefs.h>
#include <AppDemoGuiVideoControls.h>
#include <AppDemoGuiVideoStorage.h>
#include <AppDemoGuiSlamLoad.h>
#include <AppDemoGuiTestOpen.h>
#include <AppDemoGuiTestWrite.h>
#include <AppDemoGuiSlamParam.h>
#include <AppWAI.h>
#include <AppDirectories.h>
#include <AppWAISlamParamHelper.h>

AppDemoGuiAbout* WAIApp::aboutDial = nullptr;
AppDemoGuiError* WAIApp::errorDial = nullptr;

GUIPreferences     WAIApp::uiPrefs;
SLGLTexture*       WAIApp::cpvrLogo   = nullptr;
SLGLTexture*       WAIApp::videoImage = nullptr;
AppWAIDirectories* WAIApp::dirs       = nullptr;
AppWAIScene*       WAIApp::waiScene   = nullptr;

int WAIApp::liveVideoTargetWidth;
int WAIApp::liveVideoTargetHeight;
int WAIApp::trackingImgWidth;

cv::VideoWriter*   WAIApp::videoWriter     = nullptr;
cv::VideoWriter*   WAIApp::videoWriterInfo = nullptr;
WAI::ModeOrbSlam2* WAIApp::mode            = nullptr;
bool               WAIApp::loaded          = false;
ofstream           WAIApp::gpsDataStream;
SLGLTexture*       WAIApp::testTexture;
SLQuat4f           WAIApp::lastKnowPoseQuaternion;
SLQuat4f           WAIApp::IMUQuaternion;
SLVec3f            WAIApp::lastKnowPosePosition;
SLVec3f            WAIApp::IMUPosition;

SlamParams* WAIApp::currentSlamParams = nullptr;

std::string WAIApp::videoDir       = "";
std::string WAIApp::calibDir       = "";
std::string WAIApp::mapDir         = "";
std::string WAIApp::vocDir         = "";
std::string WAIApp::experimentsDir = "";

//basic information
//-the sceen has a fixed size on android and also a fixed aspect ratio on desktop to simulate android behaviour on desktop
//-the viewport gets adjusted according to the target video aspect ratio (for live video WAIApp::liveVideoTargetWidth and WAIApp::liveVideoTargetHeight are used and for video file the video frame size is used)
//-the live video gets cropped according to the viewport aspect ratio on android and according to WAIApp::videoFrameWdivH on desktop (which is also used to define the viewport aspect ratio)
//-the calibration gets adjusted according to the video (live and file)
//-the live video gets cropped to the aspect ratio that is defined by the transferred values in load(..) and assigned to liveVideoTargetWidth and liveVideoTargetHeight

float      WAIApp::videoFrameWdivH;
cv::Size2i WAIApp::videoFrameSize;

bool WAIApp::resizeWindow = false;

bool WAIApp::pauseVideo           = false;
int  WAIApp::videoCursorMoveIndex = 0;

int WAIApp::load(int liveVideoTargetW, int liveVideoTargetH, int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi, AppWAIDirectories* directories)
{
    liveVideoTargetWidth  = liveVideoTargetW;
    liveVideoTargetHeight = liveVideoTargetH;

    dirs = directories;

    SLApplication::devRot.isUsed(true);
    SLApplication::devLoc.isUsed(true);

    currentSlamParams = new SlamParams;

    videoDir       = dirs->writableDir + "erleb-AR/locations/";
    calibDir       = dirs->writableDir + "calibrations/";
    mapDir         = dirs->writableDir + "maps/";
    vocDir         = dirs->writableDir + "voc/";
    experimentsDir = dirs->writableDir + "experiments/";

    waiScene        = new AppWAIScene();
    videoWriter     = new cv::VideoWriter();
    videoWriterInfo = new cv::VideoWriter();

    SLVstring empty;
    empty.push_back("WAI APP");
    slCreateAppAndScene(empty,
                        dirs->slDataRoot + "/shaders/",
                        dirs->slDataRoot + "/models/",
                        dirs->slDataRoot + "/images/textures/",
                        dirs->slDataRoot + "/images/fonts/",
                        dirs->writableDir,
                        "WAI Demo App",
                        (void*)WAIApp::onLoadWAISceneView);

    // This load the GUI configs that are locally stored
    uiPrefs.setDPI(dpi);
    uiPrefs.load();

    int svIndex = slCreateSceneView((int)(scrWidth * scr2fbX),
                                    (int)(scrHeight * scr2fbY),
                                    dpi,
                                    (SLSceneID)0,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    (void*)buildGUI);

    loaded = true;
    SLApplication::devRot.isUsed(true);
    SLApplication::devLoc.isUsed(true);

    return svIndex;
}

void WAIApp::close()
{
    uiPrefs.save();
    //ATTENTION: Other imgui stuff is automatically saved every 5 seconds
}

/*
videoFile: path to a video or empty if live video should be used
calibrationFile: path to a calibration or empty if calibration should be searched automatically
mapFile: path to a map or empty if no map should be used
*/
OrbSlamStartResult WAIApp::startOrbSlam(SlamParams* slamParams)
{
    OrbSlamStartResult result = {};
    uiPrefs.showError         = false;

    std::string               videoFile       = "";
    std::string               calibrationFile = "";
    std::string               mapFile         = "";
    std::string               vocFile         = "";
    std::string               markerFile      = "";
    WAI::ModeOrbSlam2::Params params;

    if (slamParams)
    {
        videoFile       = slamParams->videoFile;
        calibrationFile = slamParams->calibrationFile;
        mapFile         = slamParams->mapFile;
        vocFile         = slamParams->vocabularyFile;
        markerFile      = slamParams->markerFile;
        params          = slamParams->params;
    }

    bool useVideoFile             = !videoFile.empty();
    bool detectCalibAutomatically = calibrationFile.empty();
    bool useMapFile               = !mapFile.empty();

    // reset stuff
    if (mode)
    {
        mode->requestStateIdle();
        while (!mode->hasStateIdle())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        delete mode;
        mode = nullptr;
    }

    // Check that files exist
    if (useVideoFile && !Utils::fileExists(videoFile))
    {
        result.errorString = "Video file " + videoFile + " does not exist.";
        return result;
    }

    // determine correct calibration file
    std::string calibrationFileName;
    if (detectCalibAutomatically)
    {
        std::string computerInfo;

        if (useVideoFile)
        {
            SlamVideoInfos slamVideoInfos;
            std::string    videoFileName = Utils::getFileNameWOExt(videoFile);

            if (!extractSlamVideoInfosFromFileName(videoFileName, &slamVideoInfos))
            {
                result.errorString = "Could not extract computer infos from video filename.";
                return result;
            }

            computerInfo = slamVideoInfos.deviceString;
        }
        else
        {
            computerInfo = SLApplication::getComputerInfos();
        }

        calibrationFileName = "camCalib_" + computerInfo + "_main.xml";
        calibrationFile     = calibDir + calibrationFileName;
    }
    else
    {
        calibrationFileName = Utils::getFileName(calibrationFile);
    }

    if (!Utils::fileExists(calibrationFile))
    {
        result.errorString = "Calibration file " + calibrationFile + " does not exist.";
        return result;
    }

    if (!checkCalibration(calibDir, calibrationFileName))
    {
        result.errorString = "Calibration file " + calibrationFile + " is incorrect.";
        return result;
    }

    if (vocFile.empty())
    {
        vocFile = vocDir + "ORBvoc.bin";
    }

    if (!Utils::fileExists(vocFile))
    {
        result.errorString = "Vocabulary file does not exist: " + vocFile;
        return result;
    }

    if (useMapFile && !Utils::fileExists(mapFile))
    {
        result.errorString = "Map file " + mapFile + " does not exist.";
        return result;
    }

    CVCapture* cap = CVCapture::instance();

    // 1. Initialize CVCapture with either video file or live video
    if (useVideoFile)
    {
        cap->videoType(VT_FILE);
        cap->videoFilename = videoFile;
        cap->videoLoops    = true;
        videoFrameSize     = cap->openFile();
    }
    else
    {
        cap->videoType(VT_MAIN);
        //open(0) only has an effect on desktop. On Android it just returns {0,0}
        cap->open(0);
        videoFrameSize = cv::Size2i(liveVideoTargetWidth, liveVideoTargetHeight);
    }
    videoFrameWdivH = (float)videoFrameSize.width / (float)videoFrameSize.height;

    //scrWidth  = videoFrameSize.width;
    //scrHeight = videoFrameSize.height;
    //scrWdivH  = (float)scrWidth / (float)scrHeight;

    // 2. Load Calibration
    if (!cap->activeCamera->load(calibDir, Utils::getFileName(calibrationFile), 0, 0))
    {
        result.errorString = "Error when loading calibration from file: " +
                             calibrationFile;
        return result;
    }

    if (cap->activeCamera->imageSize() != videoFrameSize)
    {
        cap->activeCamera->adaptForNewResolution(videoFrameSize);
    }

    // 3. Adjust FOV of camera node according to new calibration (fov is used in projection->prespective mode)
    waiScene->cameraNode->fov(cap->activeCamera->cameraFovVDeg());
    // Set camera intrinsics for scene camera frustum. (used in projection->intrinsics mode)
    cv::Mat scMat = cap->activeCamera->cameraMatUndistorted();
    std::cout << "scMat: " << scMat << std::endl;
    waiScene->cameraNode->intrinsics(scMat.at<double>(0, 0),
                                     scMat.at<double>(1, 1),
                                     scMat.at<double>(0, 2),
                                     scMat.at<double>(1, 2));
    //enable projection -> intrinsics mode
    waiScene->cameraNode->projection(P_monoIntrinsic);
    //enable image undistortion
    cap->activeCamera->showUndistorted(true);

    // 4. Create new mode ORBSlam
    if (!markerFile.empty())
    {
        params.cullRedundantPerc = 0.99f;
    }

    mode = new WAI::ModeOrbSlam2(cap->activeCamera->cameraMat(),
                                 cap->activeCamera->distortion(),
                                 params,
                                 vocFile,
                                 markerFile);

    // 5. Load map data
    if (useMapFile)
    {
        mode->requestStateIdle();
        while (!mode->hasStateIdle())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        mode->reset();

        // TODO(dgj1): extract feature type
        // TODO(dgj1): check that map feature type matches with mode feature type
        bool mapLoadingSuccess = WAIMapStorage::loadMap(mode->getMap(),
                                                        mode->getKfDB(),
                                                        waiScene->mapNode,
                                                        mapFile,
                                                        WAIApp::mode->retainImage(),
                                                        params.fixOldKfs);

        if (!mapLoadingSuccess)
        {
            delete mode;
            mode = nullptr;

            result.errorString = "Could not load map from file " + mapFile;
            return result;
        }

        mode->resume();
        mode->setInitialized(true);
    }

    // 6. resize window
    // resizeWindow = true;
    //ghm1: I do it outside of this function after viewport change now

    currentSlamParams->calibrationFile  = calibrationFile;
    currentSlamParams->mapFile          = mapFile;
    currentSlamParams->params.retainImg = params.retainImg;
    currentSlamParams->videoFile        = videoFile;
    currentSlamParams->vocabularyFile   = vocFile;

    result.wasSuccessful = true;

    return result;
}

bool WAIApp::checkCalibration(const std::string& calibDir, const std::string& calibFileName)
{
    CVCalibration testCalib;
    testCalib.load(calibDir, calibFileName, false, false);
    if (testCalib.cameraMat().empty() || testCalib.distortion().empty()) //app will crash if distortion is empty
        return false;
    if (testCalib.numCapturedImgs() == 0) //if this is 0 then the calibration is automatically invalidated in load()
        return false;
    if (testCalib.imageSize() == cv::Size(0, 0))
        return false;

    return true;
}

void WAIApp::setupGUI()
{
    aboutDial = new AppDemoGuiAbout("about", cpvrLogo, &uiPrefs.showAbout);
    AppDemoGui::addInfoDialog(new AppDemoGuiInfosFrameworks("frameworks", &uiPrefs.showInfosFrameworks));
    AppDemoGui::addInfoDialog(new AppDemoGuiInfosMapNodeTransform("map node",
                                                                  waiScene->mapNode,
                                                                  &uiPrefs.showInfosMapNodeTransform));

    AppDemoGui::addInfoDialog(new AppDemoGuiInfosScene("scene", &uiPrefs.showInfosScene));
    AppDemoGui::addInfoDialog(new AppDemoGuiInfosSensors("sensors", &uiPrefs.showInfosSensors));
    AppDemoGui::addInfoDialog(new AppDemoGuiInfosTracking("tracking", uiPrefs));
    AppDemoGui::addInfoDialog(new AppDemoGuiSlamLoad("slam load",
                                                     dirs->writableDir + "erleb-AR/locations/",
                                                     dirs->writableDir + "calibrations/",
                                                     dirs->writableDir + "voc/",
                                                     waiScene->mapNode,
                                                     &uiPrefs.showSlamLoad));

    AppDemoGui::addInfoDialog(new AppDemoGuiProperties("properties", &uiPrefs.showProperties));
    AppDemoGui::addInfoDialog(new AppDemoGuiSceneGraph("scene graph", &uiPrefs.showSceneGraph));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsDebugTiming("debug timing", &uiPrefs.showStatsDebugTiming));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsTiming("timing", &uiPrefs.showStatsTiming));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsVideo("video", CVCapture::instance()->activeCamera, &uiPrefs.showStatsVideo));
    AppDemoGui::addInfoDialog(new AppDemoGuiTrackedMapping("tracked mapping", &uiPrefs.showTrackedMapping));

    AppDemoGui::addInfoDialog(new AppDemoGuiTransform("transform", &uiPrefs.showTransform));
    AppDemoGui::addInfoDialog(new AppDemoGuiUIPrefs("prefs", &uiPrefs, &uiPrefs.showUIPrefs));

    AppDemoGui::addInfoDialog(new AppDemoGuiVideoStorage("video storage", videoWriter, videoWriterInfo, &gpsDataStream, &uiPrefs.showVideoStorage));
    AppDemoGui::addInfoDialog(new AppDemoGuiVideoControls("video load", &uiPrefs.showVideoControls));

    AppDemoGui::addInfoDialog(new AppDemoGuiTestOpen("Tests Settings",
                                                     waiScene->mapNode,
                                                     &uiPrefs.showTestSettings));

    AppDemoGui::addInfoDialog(new AppDemoGuiTestWrite("Test Writer",
                                                      CVCapture::instance()->activeCamera,
                                                      waiScene->mapNode,
                                                      videoWriter,
                                                      videoWriterInfo,
                                                      &gpsDataStream,
                                                      &uiPrefs.showTestWriter));

    AppDemoGui::addInfoDialog(new AppDemoGuiSlamParam("Slam Param", &uiPrefs.showSlamParam));
    errorDial = new AppDemoGuiError("Error", &uiPrefs.showError);

    AppDemoGui::addInfoDialog(errorDial);
    //TODO: AppDemoGuiInfosDialog are never deleted. Why not use smart pointer when the reponsibility for an object is not clear?
}

void WAIApp::buildGUI(SLScene* s, SLSceneView* sv)
{
    if (uiPrefs.showAbout)
    {
        aboutDial->buildInfos(s, sv);
    }
    else
    {
        AppDemoGui::buildInfosDialogs(s, sv);
        AppDemoGuiMenu::build(&uiPrefs, s, sv);
    }
}
//-----------------------------------------------------------------------------
void WAIApp::onLoadWAISceneView(SLScene* s, SLSceneView* sv, SLSceneID sid)
{
    s->init();
    waiScene->rebuild();
    //setup gui at last because ui elements depend on other instances
    setupGUI();

    // Set scene name and info string
    s->name("Track Keyframe based Features");
    s->info("Example for loading an existing pose graph with map points.");

    // Save no energy
    sv->doWaitOnIdle(false); //for constant video feed
    sv->camera(waiScene->cameraNode);

    videoImage = new SLGLTexture("LiveVideoError.png", GL_LINEAR, GL_LINEAR);
    //testTexture = new SLGLTexture("LiveVideoError.png", GL_LINEAR, GL_LINEAR);
    waiScene->cameraNode->background().texture(videoImage);
    //waiScene->cameraNode->background().texture(testTexture);

    s->root3D(waiScene->rootNode);

    sv->onInitialize();
    sv->doWaitOnIdle(false);

    /*OrbSlamStartResult orbSlamStartResult = startOrbSlam();

    if (!orbSlamStartResult.wasSuccessful)
    {
        errorDial->setErrorMsg(orbSlamStartResult.errorString);
        uiPrefs.showError = true;
    }*/

    ////setup gui at last because ui elements depend on other instances
    //setupGUI();
    sv->setViewportFromRatio(SLVec2i(WAIApp::liveVideoTargetWidth, WAIApp::liveVideoTargetHeight), SLViewportAlign::VA_center, true);
    float wdh = sv->scrWdivH();
    //do once an onResize in update loop so that everything is aligned correctly
    WAIApp::resizeWindow = true;
}

//-----------------------------------------------------------------------------
bool WAIApp::update()
{
    AVERAGE_TIMING_START("WAIAppUpdate");
    //resetAllTexture();
    if (!mode)
        return false;

    if (!loaded)
        return false;

    if (CVCapture::instance()->lastFrame.empty() ||
        CVCapture::instance()->lastFrame.cols == 0 && CVCapture::instance()->lastFrame.rows == 0)
    {
        //this only has an influence on desktop or video file
        CVCapture::instance()->grabAndAdjustForSL(videoFrameWdivH);
        return false;
    }

    bool iKnowWhereIAm = (mode->getTrackingState() == WAI::TrackingState_TrackingOK);
    while (videoCursorMoveIndex < 0)
    {
        //this only has an influence on desktop or video file
        CVCapture::instance()->moveCapturePosition(-2);
        CVCapture::instance()->grabAndAdjustForSL(videoFrameWdivH);
        iKnowWhereIAm = updateTracking();

        videoCursorMoveIndex++;
    }

    while (videoCursorMoveIndex > 0)
    {
        //this only has an influence on desktop or video file
        CVCapture::instance()->grabAndAdjustForSL(videoFrameWdivH);
        iKnowWhereIAm = updateTracking();

        videoCursorMoveIndex--;
    }

    if (CVCapture::instance()->videoType() != VT_NONE)
    {
        //this only has an influence on desktop or video file
        if (CVCapture::instance()->videoType() != VT_FILE || !pauseVideo)
        {
            CVCapture::instance()->grabAndAdjustForSL(videoFrameWdivH);
            iKnowWhereIAm = updateTracking();
        }
    }

    //update tracking infos visualization
    updateTrackingVisualization(iKnowWhereIAm);

    if (iKnowWhereIAm)
    {
        lastKnowPoseQuaternion = SLApplication::devRot.quaternion();
        IMUQuaternion          = SLQuat4f(0, 0, 0, 1);
        // TODO(dgj1): maybe make this API cleaner
        cv::Mat pose = cv::Mat(4, 4, CV_32F);
        if (!mode->getPose(&pose))
        {
            return false;
        }

        // update camera node position
        cv::Mat Rwc(3, 3, CV_32F);
        cv::Mat twc(3, 1, CV_32F);

        Rwc = (pose.rowRange(0, 3).colRange(0, 3)).t();
        twc = -Rwc * pose.rowRange(0, 3).col(3);

        cv::Mat PoseInv = cv::Mat::eye(4, 4, CV_32F);

        Rwc.copyTo(PoseInv.colRange(0, 3).rowRange(0, 3));
        twc.copyTo(PoseInv.rowRange(0, 3).col(3));
        SLMat4f om;

        om.setMatrix(PoseInv.at<float>(0, 0),
                     -PoseInv.at<float>(0, 1),
                     -PoseInv.at<float>(0, 2),
                     PoseInv.at<float>(0, 3),
                     PoseInv.at<float>(1, 0),
                     -PoseInv.at<float>(1, 1),
                     -PoseInv.at<float>(1, 2),
                     PoseInv.at<float>(1, 3),
                     PoseInv.at<float>(2, 0),
                     -PoseInv.at<float>(2, 1),
                     -PoseInv.at<float>(2, 2),
                     PoseInv.at<float>(2, 3),
                     PoseInv.at<float>(3, 0),
                     -PoseInv.at<float>(3, 1),
                     -PoseInv.at<float>(3, 2),
                     PoseInv.at<float>(3, 3));

        waiScene->cameraNode->om(om);
    }
    else
    {
        SLQuat4f q1 = lastKnowPoseQuaternion;
        SLQuat4f q2 = SLApplication::devRot.quaternion();
        q1.invert();
        SLQuat4f q     = q1 * q2;
        IMUQuaternion  = SLQuat4f(q.y(), -q.x(), -q.z(), -q.w());
        SLMat4f imuRot = IMUQuaternion.toMat4();

        SLMat4f cameraMat = waiScene->cameraNode->om();

        waiScene->cameraNode->om(cameraMat * imuRot);
        lastKnowPoseQuaternion = q2;
    }

    AVERAGE_TIMING_STOP("WAIAppUpdate");

    return true;
}
//-----------------------------------------------------------------------------
bool WAIApp::updateTracking()
{
    bool iKnowWhereIAm = false;

    CVCapture* cap = CVCapture::instance();
    if (cap->videoType() != VT_NONE && !cap->lastFrame.empty())
    {
        if (videoWriter->isOpened())
        {
            videoWriter->write(cap->lastFrame);
        }

        iKnowWhereIAm = mode->update(cap->lastFrameGray,
                                     cap->lastFrame);

        if (videoWriterInfo->isOpened())
        {
            videoWriterInfo->write(cap->lastFrame);
        }

        if (gpsDataStream.is_open())
        {
            if (SLApplication::devLoc.isUsed())
            {
                SLVec3d v = SLApplication::devLoc.locLLA();
                gpsDataStream << SLApplication::devLoc.locAccuracyM();
                gpsDataStream << std::to_string(v.x) + " " + std::to_string(v.y) + " " + std::to_string(v.z);
                gpsDataStream << std::to_string(SLApplication::devRot.yawRAD());
                gpsDataStream << std::to_string(SLApplication::devRot.pitchRAD());
                gpsDataStream << std::to_string(SLApplication::devRot.rollRAD());
            }
        }
    }

    return iKnowWhereIAm;
}
//-----------------------------------------------------------------------------
void WAIApp::updateTrackingVisualization(const bool iKnowWhereIAm)
{
    CVCapture* cap = CVCapture::instance();
    //undistort image and copy image to video texture
    if (videoImage && cap->activeCamera)
    {
        //decorate distorted image with distorted keypoints
        if (uiPrefs.showKeyPoints)
            mode->decorateVideoWithKeyPoints(cap->lastFrame);
        if (uiPrefs.showKeyPointsMatched)
            mode->decorateVideoWithKeyPointMatches(cap->lastFrame);

        CVMat undistortedLastFrame;
        if (cap->activeCamera->state() == CS_calibrated && cap->activeCamera->showUndistorted())
        {
            cap->activeCamera->remap(cap->lastFrame, undistortedLastFrame);
        }
        else
        {
            undistortedLastFrame = cap->lastFrame;
        }

        videoImage->copyVideoImage(undistortedLastFrame.cols,
                                   undistortedLastFrame.rows,
                                   cap->format,
                                   undistortedLastFrame.data,
                                   undistortedLastFrame.isContinuous(),
                                   true);
    }

    //update map point visualization:
    //if we still want to visualize the point cloud
    if (uiPrefs.showMapPC)
    {
        //get new points and add them
        renderMapPoints("MapPoints",
                        mode->getMapPoints(),
                        waiScene->mapPC,
                        waiScene->mappointsMesh,
                        waiScene->redMat);

        //get new points and add them
        renderMapPoints("MarkerCornerMapPoints",
                        mode->getMarkerCornerMapPoints(),
                        waiScene->mapMarkerCornerPC,
                        waiScene->mappointsMarkerCornerMesh,
                        waiScene->blueMat);
    }
    else
    {
        if (waiScene->mappointsMesh)
        {
            //delete mesh if we do not want do visualize it anymore
            waiScene->mapPC->deleteMesh(waiScene->mappointsMesh);
        }
        if (waiScene->mappointsMarkerCornerMesh)
        {
            //delete mesh if we do not want do visualize it anymore
            waiScene->mapMarkerCornerPC->deleteMesh(waiScene->mappointsMarkerCornerMesh);
        }
    }

    //update visualization of local map points:
    //only update them with a valid pose from WAI
    if (uiPrefs.showLocalMapPC && iKnowWhereIAm)
    {
        renderMapPoints("LocalMapPoints",
                        mode->getLocalMapPoints(),
                        waiScene->mapLocalPC,
                        waiScene->mappointsLocalMesh,
                        waiScene->blueMat);
    }
    else if (waiScene->mappointsLocalMesh)
    {
        //delete mesh if we do not want do visualize it anymore
        waiScene->mapLocalPC->deleteMesh(waiScene->mappointsLocalMesh);
    }

    //update visualization of matched map points
    //only update them with a valid pose from WAI
    if (uiPrefs.showMatchesPC && iKnowWhereIAm)
    {
        renderMapPoints("MatchedMapPoints",
                        mode->getMatchedMapPoints(),
                        waiScene->mapMatchedPC,
                        waiScene->mappointsMatchedMesh,
                        waiScene->greenMat);
    }
    else if (waiScene->mappointsMatchedMesh)
    {
        //delete mesh if we do not want do visualize it anymore
        waiScene->mapMatchedPC->deleteMesh(waiScene->mappointsMatchedMesh);
    }

    //update keyframe visualization
    waiScene->keyFrameNode->deleteChildren();
    if (uiPrefs.showKeyFrames)
    {
        renderKeyframes();
    }

    //update pose graph visualization
    renderGraphs();
}

//-----------------------------------------------------------------------------
void WAIApp::renderMapPoints(std::string                      name,
                             const std::vector<WAIMapPoint*>& pts,
                             SLNode*&                         node,
                             SLPoints*&                       mesh,
                             SLMaterial*&                     material)
{
    //remove old mesh, if it exists
    if (mesh)
        node->deleteMesh(mesh);

    //instantiate and add new mesh
    if (pts.size())
    {
        //get points as Vec3f
        std::vector<SLVec3f> points, normals;
        for (auto mapPt : pts)
        {
            WAI::V3 wP = mapPt->worldPosVec();
            WAI::V3 wN = mapPt->normalVec();
            points.push_back(SLVec3f(wP.x, wP.y, wP.z));
            normals.push_back(SLVec3f(wN.x, wN.y, wN.z));
        }

        mesh = new SLPoints(points, normals, name, material);
        node->addMesh(mesh);
        node->updateAABBRec();
    }
}
//-----------------------------------------------------------------------------
void WAIApp::renderKeyframes()
{
    std::vector<WAIKeyFrame*> keyframes = mode->getKeyFrames();

    // TODO(jan): delete keyframe textures
    for (WAIKeyFrame* kf : keyframes)
    {
        if (kf->isBad())
            continue;

        SLKeyframeCamera* cam = new SLKeyframeCamera("KeyFrame " + std::to_string(kf->mnId));
        //set background
        if (kf->getTexturePath().size())
        {
            // TODO(jan): textures are saved in a global textures vector (scene->textures)
            // and should be deleted from there. Otherwise we have a yuuuuge memory leak.
#if 0
        SLGLTexture* texture = new SLGLTexture(kf->getTexturePath());
        _kfTextures.push_back(texture);
        cam->background().texture(texture);
#endif
        }

        cv::Mat Twc = kf->getObjectMatrix();

        SLMat4f om;
        om.setMatrix(Twc.at<float>(0, 0),
                     -Twc.at<float>(0, 1),
                     -Twc.at<float>(0, 2),
                     Twc.at<float>(0, 3),
                     Twc.at<float>(1, 0),
                     -Twc.at<float>(1, 1),
                     -Twc.at<float>(1, 2),
                     Twc.at<float>(1, 3),
                     Twc.at<float>(2, 0),
                     -Twc.at<float>(2, 1),
                     -Twc.at<float>(2, 2),
                     Twc.at<float>(2, 3),
                     Twc.at<float>(3, 0),
                     -Twc.at<float>(3, 1),
                     -Twc.at<float>(3, 2),
                     Twc.at<float>(3, 3));
        //om.rotate(180, 1, 0, 0);

        cam->om(om);

        //calculate vertical field of view
        SLfloat fy     = (SLfloat)kf->fy;
        SLfloat cy     = (SLfloat)kf->cy;
        SLfloat fovDeg = 2 * (SLfloat)atan2(cy, fy) * Utils::RAD2DEG;
        cam->fov(fovDeg);
        cam->focalDist(0.11f);
        cam->clipNear(0.1f);
        cam->clipFar(1000.0f);

        waiScene->keyFrameNode->addChild(cam);
    }
}
//-----------------------------------------------------------------------------
void WAIApp::renderGraphs()
{
    std::vector<WAIKeyFrame*> kfs = mode->getKeyFrames();

    SLVVec3f covisGraphPts;
    SLVVec3f spanningTreePts;
    SLVVec3f loopEdgesPts;
    for (auto* kf : kfs)
    {
        cv::Mat Ow = kf->GetCameraCenter();

        //covisibility graph
        const std::vector<WAIKeyFrame*> vCovKFs = kf->GetBestCovisibilityKeyFrames(uiPrefs.minNumOfCovisibles);

        if (!vCovKFs.empty())
        {
            for (vector<WAIKeyFrame*>::const_iterator vit = vCovKFs.begin(), vend = vCovKFs.end(); vit != vend; vit++)
            {
                if ((*vit)->mnId < kf->mnId)
                    continue;
                cv::Mat Ow2 = (*vit)->GetCameraCenter();

                covisGraphPts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
                covisGraphPts.push_back(SLVec3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2)));
            }
        }

        //spanning tree
        WAIKeyFrame* parent = kf->GetParent();
        if (parent)
        {
            cv::Mat Owp = parent->GetCameraCenter();
            spanningTreePts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
            spanningTreePts.push_back(SLVec3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2)));
        }

        //loop edges
        std::set<WAIKeyFrame*> loopKFs = kf->GetLoopEdges();
        for (set<WAIKeyFrame*>::iterator sit = loopKFs.begin(), send = loopKFs.end(); sit != send; sit++)
        {
            if ((*sit)->mnId < kf->mnId)
                continue;
            cv::Mat Owl = (*sit)->GetCameraCenter();
            loopEdgesPts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
            loopEdgesPts.push_back(SLVec3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2)));
        }
    }

    if (waiScene->covisibilityGraphMesh)
        waiScene->covisibilityGraph->deleteMesh(waiScene->covisibilityGraphMesh);

    if (covisGraphPts.size() && uiPrefs.showCovisibilityGraph)
    {
        waiScene->covisibilityGraphMesh = new SLPolyline(covisGraphPts, false, "CovisibilityGraph", waiScene->covisibilityGraphMat);
        waiScene->covisibilityGraph->addMesh(waiScene->covisibilityGraphMesh);
        waiScene->covisibilityGraph->updateAABBRec();
    }

    if (waiScene->spanningTreeMesh)
        waiScene->spanningTree->deleteMesh(waiScene->spanningTreeMesh);

    if (spanningTreePts.size() && uiPrefs.showSpanningTree)
    {
        waiScene->spanningTreeMesh = new SLPolyline(spanningTreePts, false, "SpanningTree", waiScene->spanningTreeMat);
        waiScene->spanningTree->addMesh(waiScene->spanningTreeMesh);
        waiScene->spanningTree->updateAABBRec();
    }

    if (waiScene->loopEdgesMesh)
        waiScene->loopEdges->deleteMesh(waiScene->loopEdgesMesh);

    if (loopEdgesPts.size() && uiPrefs.showLoopEdges)
    {
        waiScene->loopEdgesMesh = new SLPolyline(loopEdgesPts, false, "LoopEdges", waiScene->loopEdgesMat);
        waiScene->loopEdges->addMesh(waiScene->loopEdgesMesh);
        waiScene->loopEdges->updateAABBRec();
    }
}
