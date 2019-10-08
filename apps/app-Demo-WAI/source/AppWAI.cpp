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
#include <AppDemoGuiMapStorage.h>
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

AppDemoGuiAbout* WAIApp::aboutDial = nullptr;
AppDemoGuiError* WAIApp::errorDial = nullptr;

GUIPreferences     WAIApp::uiPrefs;
SLGLTexture*       WAIApp::cpvrLogo   = nullptr;
SLGLTexture*       WAIApp::videoImage = nullptr;
AppWAIDirectories* WAIApp::dirs       = nullptr;
AppWAIScene*       WAIApp::waiScene   = nullptr;
WAICalibration*    WAIApp::wc         = nullptr;
int                WAIApp::scrWidth;
int                WAIApp::scrHeight;
int                WAIApp::defaultScrWidth;
int                WAIApp::defaultScrHeight;
float              WAIApp::scrWdivH;
cv::VideoWriter*   WAIApp::videoWriter     = nullptr;
cv::VideoWriter*   WAIApp::videoWriterInfo = nullptr;
WAI::ModeOrbSlam2* WAIApp::mode            = nullptr;
bool               WAIApp::loaded          = false;
ofstream           WAIApp::gpsDataStream;

std::string WAIApp::videoDir       = "";
std::string WAIApp::calibDir       = "";
std::string WAIApp::mapDir         = "";
std::string WAIApp::vocDir         = "";
std::string WAIApp::experimentsDir = "";

bool WAIApp::resizeWindow = false;

bool WAIApp::pauseVideo           = false;
int  WAIApp::videoCursorMoveIndex = 0;

int WAIApp::load(int width, int height, float scr2fbX, float scr2fbY, int dpi, AppWAIDirectories* directories)
{
    defaultScrWidth  = width;
    defaultScrHeight = height;

    dirs = directories;
    SLApplication::devRot.isUsed(true);
    SLApplication::devLoc.isUsed(true);

    videoDir       = dirs->writableDir + "videos/";
    calibDir       = dirs->writableDir + "calibrations/";
    mapDir         = dirs->writableDir + "maps/";
    vocDir         = dirs->writableDir + "voc/";
    experimentsDir = dirs->writableDir + "experiments/";

    wc              = new WAICalibration();
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

    int svIndex = slCreateSceneView((int)(width * scr2fbX),
                                    (int)(height * scr2fbY),
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
OrbSlamStartResult WAIApp::startOrbSlam(std::string videoFileName,
                                        std::string calibrationFileName,
                                        std::string mapFileName,
                                        std::string vocFileName,
                                        bool        saveVideoFrames,
                                        bool        createMarkerMap)
{
    OrbSlamStartResult result = {};
    uiPrefs.showError         = false;

    bool useVideoFile             = !videoFileName.empty();
    bool detectCalibAutomatically = calibrationFileName.empty();
    bool useMapFile               = !mapFileName.empty();

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
    std::string videoFile = videoDir + videoFileName;
    if (useVideoFile && !Utils::fileExists(videoFile))
    {
        result.errorString = "Video file " + videoFile + " does not exist.";
        return result;
    }

    // determine correct calibration file
    if (detectCalibAutomatically)
    {
        std::string computerInfo;

        if (useVideoFile)
        {
            // get calibration file name from video file name
            std::vector<std::string> stringParts;
            Utils::splitString(videoFileName, '_', stringParts);

            if (stringParts.size() < 3)
            {
                result.errorString = "Could not extract computer infos from video filename.";
                return result;
            }

            computerInfo = stringParts[1];
        }
        else
        {
            computerInfo = SLApplication::getComputerInfos();
        }

        calibrationFileName = "camCalib_" + computerInfo + "_main.xml";
    }
    std::string calibrationFile = calibDir + calibrationFileName;

    if (!Utils::fileExists(calibrationFile))
    {
        result.errorString = "Calibration file " + calibrationFile + " does not exist.";
        return result;
    }

    std::string vocFile = vocDir + vocFileName;
    if (!vocFileName.empty() && !Utils::fileExists(vocFile))
    {
        result.errorString = "Vocabulary file does not exist: " + vocFile;
        return result;
    }

    std::string mapFile = mapDir + mapFileName;
    if (useMapFile && !Utils::fileExists(mapFile))
    {
        result.errorString = "Map file " + mapFile + " does not exist.";
        return result;
    }

    // 1. Initialize CVCapture with either video file or live video
    cv::Size2i videoFrameSize;
    if (useVideoFile)
    {
        CVCapture::instance()->videoType(VT_FILE);
        CVCapture::instance()->videoFilename = videoFile;
        CVCapture::instance()->videoLoops    = true;
        videoFrameSize                       = CVCapture::instance()->openFile();
    }
    else
    {
        CVCapture::instance()->videoType(VT_MAIN);
        CVCapture::instance()->open(0);

        videoFrameSize = cv::Size2i(defaultScrWidth, defaultScrHeight);
    }

    // 2. Load Calibration
    if (!wc->loadFromFile(calibrationFile))
    {
        result.errorString = "Error when loading calibration from file: " +
                             calibrationFile;
        return result;
    }

    float videoAspectRatio = (float)videoFrameSize.width / (float)videoFrameSize.height;
    float epsilon          = 0.01f;
    if (wc->aspectRatio() > videoAspectRatio + epsilon ||
        wc->aspectRatio() < videoAspectRatio - epsilon)
    {
        result.errorString = "Calibration aspect ratio does not fit video aspect ratio.\nCalib file: " +
                             calibrationFile + "\nVideo file: " +
                             (!videoFile.empty() ? videoFile : "Live Video");
        return result;
    }

    CVCapture::instance()->activeCalib->load(calibDir, calibrationFileName, 0, 0);

    // 3. Adjust FOV of camera node according to new calibration
    waiScene->cameraNode->fov(wc->calcCameraVerticalFOV());

    // 4. Create new mode ORBSlam
    mode = new WAI::ModeOrbSlam2(wc->cameraMat(),
                                 wc->distortion(),
                                 false,
                                 saveVideoFrames,
                                 false,
                                 false,
                                 createMarkerMap,
                                 vocFile);

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
                                                        mapFile);

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
    scrWidth     = videoFrameSize.width;
    scrHeight    = videoFrameSize.height;
    scrWdivH     = (float)scrWidth / (float)scrHeight;
    resizeWindow = true;

    result.wasSuccessful = true;
    return result;
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
    AppDemoGui::addInfoDialog(new AppDemoGuiSlamLoad("slam load", wc, &uiPrefs.showSlamLoad));

    AppDemoGui::addInfoDialog(new AppDemoGuiProperties("properties", &uiPrefs.showProperties));
    AppDemoGui::addInfoDialog(new AppDemoGuiSceneGraph("scene graph", &uiPrefs.showSceneGraph));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsDebugTiming("debug timing", &uiPrefs.showStatsDebugTiming));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsTiming("timing", &uiPrefs.showStatsTiming));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsVideo("video", wc, &uiPrefs.showStatsVideo));
    AppDemoGui::addInfoDialog(new AppDemoGuiTrackedMapping("tracked mapping", &uiPrefs.showTrackedMapping));

    AppDemoGui::addInfoDialog(new AppDemoGuiTransform("transform", &uiPrefs.showTransform));
    AppDemoGui::addInfoDialog(new AppDemoGuiUIPrefs("prefs", &uiPrefs, &uiPrefs.showUIPrefs));

    AppDemoGui::addInfoDialog(new AppDemoGuiVideoStorage("video storage", videoWriter, videoWriterInfo, &gpsDataStream, &uiPrefs.showVideoStorage));
    AppDemoGui::addInfoDialog(new AppDemoGuiVideoControls("video load", &uiPrefs.showVideoControls));

    AppDemoGui::addInfoDialog(new AppDemoGuiMapStorage("Map storage", waiScene->mapNode, &uiPrefs.showMapStorage));

    AppDemoGui::addInfoDialog(new AppDemoGuiTestOpen("Tests Settings",
                                                     wc,
                                                     waiScene->mapNode,
                                                     &uiPrefs.showTestSettings));

    AppDemoGui::addInfoDialog(new AppDemoGuiTestWrite("Test Writer",
                                                      wc,
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
    waiScene->cameraNode->background().texture(videoImage);

    s->root3D(waiScene->rootNode);

    sv->onInitialize();
    sv->doWaitOnIdle(false);

    OrbSlamStartResult orbSlamStartResult = startOrbSlam();

    if (!orbSlamStartResult.wasSuccessful)
    {
        errorDial->setErrorMsg(orbSlamStartResult.errorString);
        uiPrefs.showError = true;
    }
}

//-----------------------------------------------------------------------------
bool WAIApp::update()
{
    AVERAGE_TIMING_START("WAIAppUpdate");
    if (!mode)
        return false;

    if (!loaded)
        return false;

    bool iKnowWhereIAm = (mode->getTrackingState() == WAI::TrackingState_TrackingOK);
    while (videoCursorMoveIndex < 0)
    {
        CVCapture::instance()->moveCapturePosition(-2);
        CVCapture::instance()->grabAndAdjustForSL(scrWdivH);
        iKnowWhereIAm = updateTracking();

        videoCursorMoveIndex++;
    }

    while (videoCursorMoveIndex > 0)
    {
        CVCapture::instance()->grabAndAdjustForSL(scrWdivH);
        iKnowWhereIAm = updateTracking();

        videoCursorMoveIndex--;
    }

    if (CVCapture::instance()->videoType() != VT_NONE)
    {
        if (CVCapture::instance()->videoType() != VT_FILE || !pauseVideo)
        {
            CVCapture::instance()->grabAndAdjustForSL(scrWdivH);
            iKnowWhereIAm = updateTracking();
        }
    }

    //update tracking infos visualization
    updateTrackingVisualization(iKnowWhereIAm);

    if (iKnowWhereIAm)
    {
        // TODO(dgj1): maybe make this API cleaner
        cv::Mat pose = cv::Mat(4, 4, CV_32F);
        if (!mode->getPose(&pose))
        {
            return false;
        }

        cv::Mat markerCorrectionTransformation = cv::Mat(3, 4, CV_32F);
        if (mode->getMarkerCorrectionTransformation(&markerCorrectionTransformation))
        {
            markerCorrectionTransformation.at<float>(2, 2) = 1.0f;
            std::cout << markerCorrectionTransformation << std::endl;

            // update camera node position
            cv::Mat Rwc(3, 3, CV_32F);
            cv::Mat twc(3, 1, CV_32F);

            Rwc = (markerCorrectionTransformation.rowRange(0, 3).colRange(0, 3)).t();
            twc = -Rwc * markerCorrectionTransformation.rowRange(0, 3).col(3);

            cv::Mat CorInv = cv::Mat::eye(4, 4, CV_32F);

            Rwc.copyTo(CorInv.colRange(0, 3).rowRange(0, 3));
            twc.copyTo(CorInv.rowRange(0, 3).col(3));
            SLMat4f nodeOm;

            nodeOm.setMatrix(CorInv.at<float>(0, 0),
                             -CorInv.at<float>(0, 1),
                             -CorInv.at<float>(0, 2),
                             CorInv.at<float>(0, 3),
                             CorInv.at<float>(1, 0),
                             -CorInv.at<float>(1, 1),
                             -CorInv.at<float>(1, 2),
                             CorInv.at<float>(1, 3),
                             CorInv.at<float>(2, 0),
                             -CorInv.at<float>(2, 1),
                             -CorInv.at<float>(2, 2),
                             CorInv.at<float>(2, 3),
                             CorInv.at<float>(3, 0),
                             -CorInv.at<float>(3, 1),
                             -CorInv.at<float>(3, 2),
                             CorInv.at<float>(3, 3));

            waiScene->mapNode->om(nodeOm);
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

    AVERAGE_TIMING_STOP("WAIAppUpdate");

    return true;
}
//-----------------------------------------------------------------------------
bool WAIApp::updateTracking()
{
    bool iKnowWhereIAm = false;

    if (CVCapture::instance()->videoType() != VT_NONE && !CVCapture::instance()->lastFrame.empty())
    {
        if (videoWriter->isOpened())
        {
            videoWriter->write(CVCapture::instance()->lastFrame);
        }

        iKnowWhereIAm = mode->update(CVCapture::instance()->lastFrameGray,
                                     CVCapture::instance()->lastFrame);

        if (videoWriterInfo->isOpened())
        {
            videoWriterInfo->write(CVCapture::instance()->lastFrame);
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
    // refresh video image
    videoImage->copyVideoImage(CVCapture::instance()->lastFrame.cols,
                               CVCapture::instance()->lastFrame.rows,
                               CVCapture::instance()->format,
                               CVCapture::instance()->lastFrame.data,
                               CVCapture::instance()->lastFrame.isContinuous(),
                               true);

    //update keypoints visualization (2d image points):
    //TODO: 2d visualization is still done in mode... do we want to keep it there?
    mode->showKeyPoints(uiPrefs.showKeyPoints);
    mode->showKeyPointsMatched(uiPrefs.showKeyPointsMatched);

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
    }
    else if (waiScene->mappointsMesh)
    {
        //delete mesh if we do not want do visualize it anymore
        waiScene->mapPC->deleteMesh(waiScene->mappointsMesh);
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
