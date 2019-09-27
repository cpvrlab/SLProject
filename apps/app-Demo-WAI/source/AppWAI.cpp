#include <atomic>
#include <SLApplication.h>
#include <SLInterface.h>
#include <SLKeyframeCamera.h>
#include <CVCapture.h>
#include <Utils.h>

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

int   WAIApp::minNumOfCovisibles = 50;
float WAIApp::meanReprojectionError;
bool  WAIApp::showKeyPoints         = true;
bool  WAIApp::showKeyPointsMatched  = true;
bool  WAIApp::showMapPC             = true;
bool  WAIApp::showLocalMapPC        = true;
bool  WAIApp::showMatchesPC         = true;
bool  WAIApp::showKeyFrames         = true;
bool  WAIApp::renderKfBackground    = true;
bool  WAIApp::allowKfsAsActiveCam   = true;
bool  WAIApp::showCovisibilityGraph = true;
bool  WAIApp::showSpanningTree      = true;
bool  WAIApp::showLoopEdges         = true;

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
float              WAIApp::scrWdivH;
cv::VideoWriter*   WAIApp::videoWriter     = nullptr;
cv::VideoWriter*   WAIApp::videoWriterInfo = nullptr;
WAI::ModeOrbSlam2* WAIApp::mode            = nullptr;
bool               WAIApp::loaded          = false;
ofstream           WAIApp::gpsDataStream;

bool WAIApp::resizeWindow = false;

bool WAIApp::pauseVideo = false;

int WAIApp::load(int width, int height, float scr2fbX, float scr2fbY, int dpi, AppWAIDirectories* directories)
{
    scrWidth  = width;
    scrHeight = height;
    scrWdivH  = (float)width / (float)height;

    dirs = directories;
    SLApplication::devRot.isUsed(true);
    SLApplication::devLoc.isUsed(true);

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

/*
videoFile: path to a video or empty if live video should be used
calibrationFile: path to a calibration or empty if calibration should be searched automatically
mapFile: path to a map or empty if no map should be used
*/
OrbSlamStartResult WAIApp::startOrbSlam(std::string videoFile,
                                        std::string calibrationFile,
                                        std::string mapFile)
{
    OrbSlamStartResult result = {};

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
    if (!videoFile.empty() && !Utils::fileExists(videoFile))
    {
        result.errorString = "Video file " + videoFile + "does not exist.";
        return result;
    }

    if (!calibrationFile.empty() && !Utils::fileExists(calibrationFile))
    {
        result.errorString = "Calibration file " + calibrationFile + "does not exist.";
        return result;
    }

    if (!mapFile.empty() && !Utils::fileExists(mapFile))
    {
        result.errorString = "Map file " + mapFile + "does not exist.";
        return result;
    }

    // 1. Initialize CVCapture with either video file or live video
    cv::Size2i videoFrameSize;
    if (!videoFile.empty())
    {
        CVCapture::instance()->videoType(VT_FILE);
        CVCapture::instance()->videoFilename = videoFile;
        CVCapture::instance()->videoLoops    = true;
        videoFrameSize                       = CVCapture::instance()->openFile();
    }
    else
    {
        // TODO(dgj1): adjust scrWdivH according to current screen size?
        CVCapture::instance()->videoType(VT_MAIN);
        CVCapture::instance()->open(0);

        videoFrameSize = cv::Size2i(640, 360); // TODO(dgj1): rethink this fixed 16:9 aspect ratio
    }

    // 2. Load Calibration
    if (!calibrationFile.empty())
    {
        wc->loadFromFile(calibrationFile);
    }
    else
    {
        std::string computerInfo;

        if (!videoFile.empty())
        {
            // get calibration file name from video file name
            std::vector<std::string> stringParts;
            Utils::splitString(Utils::getFileNameWOExt(videoFile), '_', stringParts);

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

        calibrationFile = dirs->writableDir + "calibrations/camCalib_" + computerInfo + "_main.xml";
        wc->loadFromFile(calibrationFile);
    }

    float videoAspectRatio = (float)videoFrameSize.width / (float)videoFrameSize.height;
    float epsilon          = 0.0001f;
    if (wc->aspectRatio() > videoAspectRatio + epsilon ||
        wc->aspectRatio() < videoAspectRatio - epsilon)
    {
        result.errorString = "Calibration aspect ratio does not fit video aspect ratio.\nCalib file: " +
                             calibrationFile + "\nVideo file: " +
                             (!videoFile.empty() ? videoFile : "Live Video");
        return result;
    }

    // 3. Adjust FOV of camera node according to new calibration
    waiScene->cameraNode->fov(wc->calcCameraVerticalFOV());

    // 4. Create new mode ORBSlam
    mode = new WAI::ModeOrbSlam2(wc->cameraMat(),
                                 wc->distortion(),
                                 false,
                                 false,
                                 false,
                                 false,
                                 dirs->slDataRoot + "/calibrations/ORBvoc.bin");

    // 5. Load map data
    if (!mapFile.empty())
    {
        mode->requestStateIdle();
        while (!mode->hasStateIdle())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        mode->reset();

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
                                                                  dirs->writableDir,
                                                                  &uiPrefs.showInfosMapNodeTransform));

    AppDemoGui::addInfoDialog(new AppDemoGuiInfosScene("scene", &uiPrefs.showInfosScene));
    AppDemoGui::addInfoDialog(new AppDemoGuiInfosSensors("sensors", &uiPrefs.showInfosSensors));
    AppDemoGui::addInfoDialog(new AppDemoGuiInfosTracking("tracking", mode, &uiPrefs.showInfosTracking));
    AppDemoGui::addInfoDialog(new AppDemoGuiSlamLoad("slam load", dirs->writableDir + "videos/", dirs->writableDir + "calibrations/", dirs->writableDir + "maps/", wc, &uiPrefs.showSlamLoad));

    AppDemoGui::addInfoDialog(new AppDemoGuiProperties("properties", &uiPrefs.showProperties));
    AppDemoGui::addInfoDialog(new AppDemoGuiSceneGraph("scene graph", &uiPrefs.showSceneGraph));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsDebugTiming("debug timing", &uiPrefs.showStatsDebugTiming));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsTiming("timing", &uiPrefs.showStatsTiming));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsVideo("video", wc, &uiPrefs.showStatsVideo));
    AppDemoGui::addInfoDialog(new AppDemoGuiTrackedMapping("tracked mapping", &uiPrefs.showTrackedMapping));

    AppDemoGui::addInfoDialog(new AppDemoGuiTransform("transform", &uiPrefs.showTransform));
    AppDemoGui::addInfoDialog(new AppDemoGuiUIPrefs("prefs", &uiPrefs, &uiPrefs.showUIPrefs));

    AppDemoGui::addInfoDialog(new AppDemoGuiVideoStorage("video storage", dirs->writableDir + "videos/", videoWriter, videoWriterInfo, &gpsDataStream, &uiPrefs.showVideoStorage));
    AppDemoGui::addInfoDialog(new AppDemoGuiVideoControls("video load", &uiPrefs.showVideoControls));

    AppDemoGui::addInfoDialog(new AppDemoGuiMapStorage("Map storage", waiScene->mapNode, dirs->writableDir + "/maps/", &uiPrefs.showMapStorage));

    AppDemoGui::addInfoDialog(new AppDemoGuiTestOpen("Tests Settings",
                                                     dirs->writableDir + "/savedTests/",
                                                     wc,
                                                     waiScene->mapNode,
                                                     &uiPrefs.showTestSettings));

    AppDemoGui::addInfoDialog(new AppDemoGuiTestWrite("Test Writer",
                                                      dirs->writableDir + "/savedTests/",
                                                      wc,
                                                      waiScene->mapNode,
                                                      videoWriter,
                                                      videoWriterInfo,
                                                      &gpsDataStream,
                                                      &uiPrefs.showTestWriter));

    AppDemoGui::addInfoDialog(new AppDemoGuiSlamParam("Slam Param", dirs->writableDir + "/voc/", &uiPrefs.showSlamParam));
    errorDial = new AppDemoGuiError("Error", &uiPrefs.showError);

    AppDemoGui::addInfoDialog(errorDial);

    //AppDemoGui::addInfoDialog(new AppDemoGuiCalibrationLoad("Calibration Load", dirs->writableDir + "calibrations/", wai, wc, &uiPrefs.showCalibrationLoad));
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

void WAIApp::refreshTexture(cv::Mat* image)
{
    if (image == nullptr)
        return;

    videoImage->copyVideoImage(image->cols, image->rows, CVCapture::instance()->format, image->data, image->isContinuous(), true);
}

//-----------------------------------------------------------------------------
void WAIApp::onLoadWAISceneView(SLScene* s, SLSceneView* sv, SLSceneID sid)
{
    s->init();
    waiScene->rebuild();
    setupGUI();
    // Set scene name and info string
    s->name("Track Keyframe based Features");
    s->info("Example for loading an existing pose graph with map points.");

    // Save no energy
    sv->doWaitOnIdle(false); //for constant video feed
    sv->camera(waiScene->cameraNode);

    videoImage = new SLGLTexture("LiveVideoError.png", GL_LINEAR, GL_LINEAR);
    waiScene->cameraNode->background().texture(videoImage);

    //waiScene->cameraNode->fov(wc->calcCameraVerticalFOV());

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
    if (!mode)
        return false;

    if (!loaded)
        return false;

    if (CVCapture::instance()->videoType() == VT_FILE && pauseVideo)
        return true;

    if (CVCapture::instance()->videoType() != VT_NONE)
    {
        CVCapture::instance()->grabAndAdjustForSL(scrWdivH);
    }

    bool iKnowWhereIAm = false;
    if (CVCapture::instance()->videoType() != VT_NONE && !CVCapture::instance()->lastFrame.empty())
    {
        if (videoWriter->isOpened())
        {
            videoWriter->write(CVCapture::instance()->lastFrame);
        }

        iKnowWhereIAm = mode->update(CVCapture::instance()->lastFrameGray,
                                     CVCapture::instance()->lastFrame);

        videoImage->copyVideoImage(CVCapture::instance()->lastFrame.cols,
                                   CVCapture::instance()->lastFrame.rows,
                                   CVCapture::instance()->format,
                                   CVCapture::instance()->lastFrame.data,
                                   CVCapture::instance()->lastFrame.isContinuous(),
                                   true);

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

    return true;
}
//-----------------------------------------------------------------------------
void WAIApp::updateTrackingVisualization(const bool iKnowWhereIAm)
{
    //update keypoints visualization (2d image points):
    //TODO: 2d visualization is still done in mode... do we want to keep it there?
    mode->showKeyPoints(showKeyPoints);
    mode->showKeyPointsMatched(showKeyPointsMatched);

    //update map point visualization:
    //if we still want to visualize the point cloud
    if (showMapPC)
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
    if (showLocalMapPC && iKnowWhereIAm)
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
    if (showMatchesPC && iKnowWhereIAm)
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
    if (showKeyFrames)
    {
        renderKeyframes();
    }

    //update pose graph visualization
    renderGraphs();
}

//-----------------------------------------------------------------------------
void WAIApp::updateMinNumOfCovisibles(int n)
{
    minNumOfCovisibles = n;
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
        const std::vector<WAIKeyFrame*> vCovKFs = kf->GetBestCovisibilityKeyFrames(minNumOfCovisibles);

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

    if (covisGraphPts.size() && showCovisibilityGraph)
    {
        waiScene->covisibilityGraphMesh = new SLPolyline(covisGraphPts, false, "CovisibilityGraph", waiScene->covisibilityGraphMat);
        waiScene->covisibilityGraph->addMesh(waiScene->covisibilityGraphMesh);
        waiScene->covisibilityGraph->updateAABBRec();
    }

    if (waiScene->spanningTreeMesh)
        waiScene->spanningTree->deleteMesh(waiScene->spanningTreeMesh);

    if (spanningTreePts.size() && showSpanningTree)
    {
        waiScene->spanningTreeMesh = new SLPolyline(spanningTreePts, false, "SpanningTree", waiScene->spanningTreeMat);
        waiScene->spanningTree->addMesh(waiScene->spanningTreeMesh);
        waiScene->spanningTree->updateAABBRec();
    }

    if (waiScene->loopEdgesMesh)
        waiScene->loopEdges->deleteMesh(waiScene->loopEdgesMesh);

    if (loopEdgesPts.size() && showLoopEdges)
    {
        waiScene->loopEdgesMesh = new SLPolyline(loopEdgesPts, false, "LoopEdges", waiScene->loopEdgesMat);
        waiScene->loopEdges->addMesh(waiScene->loopEdgesMesh);
        waiScene->loopEdges->updateAABBRec();
    }
}
