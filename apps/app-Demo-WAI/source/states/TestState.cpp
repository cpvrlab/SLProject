#include <states/TestState.h>
#include <WAISlam.h>
#include <WAIEvent.h>
#include <SENSCamera.h>
#include <WAIMapStorage.h>
#include <AppWAISlamParamHelper.h>
#include <FtpUtils.h>

TestState::TestState(SLInputManager& inputManager,
                     SENSCamera*     camera,
                     int             screenWidth,
                     int             screenHeight,
                     int             dotsPerInch,
                     std::string     fontPath,
                     std::string     configDir,
                     std::string     vocabularyDir,
                     std::string     calibDir,
                     std::string     videoDir)
  : _gui("TestScene", configDir, dotsPerInch, fontPath),
    _s("TestScene", inputManager),
    _sv(&_s, dotsPerInch),
    _camera(camera),
    _configDir(configDir),
    _vocabularyDir(vocabularyDir),
    _calibDir(calibDir),
    _videoDir(videoDir)
{
    _sv.init("TestSceneView", screenWidth, screenHeight, nullptr, nullptr, &_gui, _configDir);
    _s.init();

    _sv.onInitialize();

    setupDefaultErlebARDirTo(_configDir);
    setupGUI();
    tryLoadLastSlam();
}

TestState::~TestState()
{
    if (_mode)
    {
        delete _mode;
        _mode = nullptr;

        _currentSlamParams.save(_configDir + "SlamParams.json");
    }
}

bool TestState::update()
{
    handleEvents();

    SENSFramePtr frame;
    if (_videoFileStream)
    {
        updateVideoTracking();
        if (!_pauseVideo)
            frame = _videoFileStream->grabNextFrame();
    }
    else if (_camera)
        frame = _camera->getLatestFrame();
    else
        Utils::log("WAI WARN", "TestState::update: No active camera or video stream available!");

    if (frame)
    {
        if (_videoWriter && _videoWriter->isOpened())
            _videoWriter->write(frame->imgRGB);

        if (_mode)
        {
            bool iKnowWhereIAm = _mode->update(frame->imgGray);
            if (iKnowWhereIAm)
                _s.updateCameraPose(_mode->getPose());

            updateTrackingVisualization(iKnowWhereIAm, frame->imgRGB);
        }
    }

    _s.onUpdate();
    return _sv.onPaint();
}

void TestState::doStart()
{
    _started = true;
}

void TestState::tryLoadLastSlam()
{
    if (_currentSlamParams.load(_configDir + "SlamParams.json"))
    {
        loadWAISceneView(_currentSlamParams.location, _currentSlamParams.area);
        startOrbSlam(_currentSlamParams);
        _guiSlamLoad->setSlamParams(_currentSlamParams);
        _gui.uiPrefs->showSlamLoad = false;
    }
    else
    {
        _gui.uiPrefs->showSlamLoad = true;
    }
}

void TestState::setupGUI()
{
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiInfosFrameworks>("frameworks", &_gui.uiPrefs->showInfosFrameworks));
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiInfosMapNodeTransform>("map node",
    //                                                                          &_gui.uiPrefs->showInfosMapNodeTransform,
    //                                                                          &_eventQueue));
    //
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiInfosScene>("scene", &_gui.uiPrefs->showInfosScene));
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiInfosSensors>("sensors", &_gui.uiPrefs->showInfosSensors));
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiInfosTracking>("tracking", *_gui.uiPrefs.get(), *this));
    //
    _guiSlamLoad = std::make_shared<AppDemoGuiSlamLoad>("slam load",
                                                        &_eventQueue,
                                                        _configDir + "erleb-AR/locations/",
                                                        _configDir + "calibrations/",
                                                        _vocabularyDir,
                                                        _featureExtractorFactory.getExtractorIdToNames(),
                                                        &_gui.uiPrefs->showSlamLoad);
    _gui.addInfoDialog(_guiSlamLoad);
    //
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiProperties>("properties", &_gui.uiPrefs->showProperties));
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiSceneGraph>("scene graph", &_gui.uiPrefs->showSceneGraph));
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiStatsDebugTiming>("debug timing", &_gui.uiPrefs->showStatsDebugTiming));
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiStatsTiming>("timing", &_gui.uiPrefs->showStatsTiming));
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiStatsVideo>("video", &_gui.uiPrefs->showStatsVideo, *this));
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiTrackedMapping>("tracked mapping", &_gui.uiPrefs->showTrackedMapping, *this));
    //
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiTransform>("transform", &_gui.uiPrefs->showTransform));
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiUIPrefs>("prefs", _gui.uiPrefs.get(), &_gui.uiPrefs->showUIPrefs));
    //
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiVideoStorage>("video/gps storage", &_gui.uiPrefs->showVideoStorage, &_eventQueue, *this));
    //    _gui.addInfoDialog(std::make_shared<AppDemoGuiVideoControls>("video load", &_gui.uiPrefs->showVideoControls, &_eventQueue, *this));
    //
    _errorDial = std::make_shared<AppDemoGuiError>("Error", &_gui.uiPrefs->showError);
    _gui.addInfoDialog(_errorDial);
}

void TestState::handleEvents()
{
    while (!_eventQueue.empty())
    {
        WAIEvent* event = _eventQueue.front();
        _eventQueue.pop();

        switch (event->type)
        {
            case WAIEventType_StartOrbSlam: {
                WAIEventStartOrbSlam* startOrbSlamEvent = (WAIEventStartOrbSlam*)event;
                loadWAISceneView(startOrbSlamEvent->params.location, startOrbSlamEvent->params.area);
                startOrbSlam(startOrbSlamEvent->params);

                delete startOrbSlamEvent;
            }
            break;

            case WAIEventType_SaveMap: {
                WAIEventSaveMap* saveMapEvent = (WAIEventSaveMap*)event;
                saveMap(saveMapEvent->location, saveMapEvent->area, saveMapEvent->marker);

                delete saveMapEvent;
            }
            break;

            case WAIEventType_VideoControl: {
                WAIEventVideoControl* videoControlEvent = (WAIEventVideoControl*)event;
                _pauseVideo                             = videoControlEvent->pauseVideo;
                _videoCursorMoveIndex                   = videoControlEvent->videoCursorMoveIndex;

                delete videoControlEvent;
            }
            break;

            case WAIEventType_VideoRecording: {
                WAIEventVideoRecording* videoRecordingEvent = (WAIEventVideoRecording*)event;

                //if videoWriter is opened we assume that recording was started before
                if (_videoWriter && _videoWriter->isOpened() /*|| _gpsDataStream.is_open()*/)
                {
                    if (_videoWriter && _videoWriter->isOpened())
                        _videoWriter->release();
                    //if (_gpsDataStream.is_open())
                    //    _gpsDataStream.close();
                }
                else
                {
                    saveVideo(videoRecordingEvent->filename);
                    //saveGPSData(videoRecordingEvent->filename);
                }

                delete videoRecordingEvent;
            }
            break;

            case WAIEventType_MapNodeTransform: {
                WAIEventMapNodeTransform* mapNodeTransformEvent = (WAIEventMapNodeTransform*)event;

                transformMapNode(mapNodeTransformEvent->tSpace, mapNodeTransformEvent->rotation, mapNodeTransformEvent->translation, mapNodeTransformEvent->scale);

                delete mapNodeTransformEvent;
            }
            break;

            case WAIEventType_DownloadCalibrationFiles: {
                WAIEventDownloadCalibrationFiles* downloadEvent = (WAIEventDownloadCalibrationFiles*)event;
                delete downloadEvent;
                downloadCalibrationFilesTo(_calibDir);
            }
            break;

            case WAIEventType_AdjustTransparency: {
                WAIEventAdjustTransparency* adjustTransparencyEvent = (WAIEventAdjustTransparency*)event;
                _s.adjustAugmentationTransparency(adjustTransparencyEvent->kt);

                delete adjustTransparencyEvent;
            }
            break;

            case WAIEventType_None:
            default: {
            }
            break;
        }
    }
}

void TestState::loadWAISceneView(std::string location, std::string area)
{
    _s.rebuild(location, area);

    _sv.doWaitOnIdle(false);
    _sv.camera(_s.cameraNode);
    _sv.onInitialize();
    if (_camera)
        _sv.setViewportFromRatio(SLVec2i(_camera->getFrameSize().width, _camera->getFrameSize().height), SLViewportAlign::VA_center, true);
}

void TestState::saveMap(std::string location,
                        std::string area,
                        std::string marker)
{
    _mode->requestStateIdle();

    std::string slamRootDir = _configDir + "erleb-AR/locations/";
    std::string mapDir      = constructSlamMapDir(slamRootDir, location, area);
    if (!Utils::dirExists(mapDir))
        Utils::makeDir(mapDir);

    std::string filename = constructSlamMapFileName(location, area, _mode->getKPextractor()->GetName());
    std::string imgDir   = constructSlamMapImgDir(mapDir, filename);

    if (_mode->retainImage())
    {
        if (!Utils::dirExists(imgDir))
            Utils::makeDir(imgDir);
    }

    if (!marker.empty())
    {
        ORBVocabulary* voc = new ORB_SLAM2::ORBVocabulary();
        voc->loadFromBinaryFile(_currentSlamParams.vocabularyFile);

        cv::Mat nodeTransform;
        if (!WAISlamTools::doMarkerMapPreprocessing(constructSlamMarkerDir(slamRootDir, location, area) + marker,
                                                    nodeTransform,
                                                    0.75f,
                                                    _mode->getKPextractor(),
                                                    _mode->getMap(),
                                                    _calibration.cameraMat(),
                                                    voc))
        {
            showErrorMsg("Failed to do marker map preprocessing");
        }
        else
        {
            std::cout << "nodeTransform: " << nodeTransform << std::endl;
            //_s.mapNode->om(WAIMapStorage::convertToSLMat(nodeTransform));
            if (!WAIMapStorage::saveMap(_mode->getMap(),
                                        _s.mapNode,
                                        mapDir + filename,
                                        imgDir))
            {
                showErrorMsg("Failed to save map " + mapDir + filename);
            }
        }
    }
    else
    {
        if (!WAIMapStorage::saveMap(_mode->getMap(),
                                    _s.mapNode,
                                    mapDir + filename,
                                    imgDir))
        {
            showErrorMsg("Failed to save map " + mapDir + filename);
        }
    }

    _mode->resume();
}

void TestState::saveVideo(std::string filename)
{
    std::string infoDir  = _videoDir + "info/";
    std::string infoPath = infoDir + filename;
    std::string path     = _videoDir + filename;

    if (!Utils::dirExists(_videoDir))
    {
        Utils::makeDir(_videoDir);
    }
    else
    {
        if (Utils::fileExists(path))
        {
            Utils::deleteFile(path);
        }
    }

    if (!Utils::dirExists(infoDir))
    {
        Utils::makeDir(infoDir);
    }
    else
    {
        if (Utils::fileExists(infoPath))
        {
            Utils::deleteFile(infoPath);
        }
    }

    if (!_videoWriter)
        _videoWriter = new cv::VideoWriter();
    else if (_videoWriter->isOpened())
        _videoWriter->release();

    bool ret = false;
    if (_videoFileStream)
        ret = _videoWriter->open(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, _videoFileStream->getFrameSize(), true);
    else if (_camera)
        ret = _videoWriter->open(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, _camera->getFrameSize(), true);
    else
        Utils::log("WAI WARN", "WAIApp::saveVideo: No active video stream or camera available!");
}

void TestState::showErrorMsg(std::string msg)
{
    assert(_errorDial && "errorDial is not initialized");

    _errorDial->setErrorMsg(msg);
    _gui.uiPrefs->showError = true;
}

/*
videoFile: path to a video or empty if live video should be used
calibrationFile: path to a calibration or empty if calibration should be searched automatically
mapFile: path to a map or empty if no map should be used
*/
void TestState::startOrbSlam(SlamParams slamParams)
{
    _errorDial->setErrorMsg("");
    _gui.uiPrefs->showError = false;
    _lastFrameIdx           = 0;
    _doubleBufferedOutput   = false;
    if (_videoFileStream)
        _videoFileStream.release();

    bool useVideoFile             = !slamParams.videoFile.empty();
    bool detectCalibAutomatically = slamParams.calibrationFile.empty();
    bool useMapFile               = !slamParams.mapFile.empty();

    // reset stuff
    if (_mode)
    {
        _mode->requestStateIdle();
        while (!_mode->hasStateIdle())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        delete _mode;
        _mode = nullptr;
    }

    // Check that files exist
    if (useVideoFile && !Utils::fileExists(slamParams.videoFile))
    {
        showErrorMsg("Video file " + slamParams.videoFile + " does not exist.");
        return;
    }

    // determine correct calibration file
    std::string calibrationFileName;
    if (detectCalibAutomatically)
    {
        std::string computerInfo;

        if (useVideoFile)
        {
            SlamVideoInfos slamVideoInfos;
            std::string    videoFileName = Utils::getFileNameWOExt(slamParams.videoFile);

            if (!extractSlamVideoInfosFromFileName(videoFileName, &slamVideoInfos))
            {
                showErrorMsg("Could not extract computer infos from video filename.");
                return;
            }

            computerInfo = slamVideoInfos.deviceString;
        }
        else
        {
            computerInfo = Utils::ComputerInfos::get();
        }

        calibrationFileName        = "camCalib_" + computerInfo + "_main.xml";
        slamParams.calibrationFile = _calibDir + calibrationFileName;
    }
    else
    {
        calibrationFileName = Utils::getFileName(slamParams.calibrationFile);
    }

    if (!Utils::fileExists(slamParams.calibrationFile))
    {
        showErrorMsg("Calibration file " + slamParams.calibrationFile + " does not exist.");
        return;
    }

    /*
    if (!checkCalibration(calibDir, calibrationFileName))
    {
        showErrorMsg("Calibration file " + calibrationFile + " is incorrect.");
        return;
    }
     */

    if (slamParams.vocabularyFile.empty())
    {
        showErrorMsg("Select a vocabulary file!");
        return;
    }

    if (!Utils::fileExists(slamParams.vocabularyFile))
    {
        showErrorMsg("Vocabulary file does not exist: " + slamParams.vocabularyFile);
        return;
    }

    if (useMapFile && !Utils::fileExists(slamParams.mapFile))
    {
        showErrorMsg("Map file " + slamParams.mapFile + " does not exist.");
        return;
    }

    // 1. Initialize video stream
    if (useVideoFile)
    {
        _videoFileStream = std::make_unique<SENSVideoStream>(slamParams.videoFile, true, false, false);
        _videoFrameSize  = _videoFileStream->getFrameSize();
    }
    else
    {
        if (!_camera)
        {
            showErrorMsg("Camera pointer is not set!");
            return;
        }
        _videoFrameSize = cv::Size2i(_camera->getFrameSize().width, _camera->getFrameSize().height);
    }

    // 2. Load Calibration
    //build undistortion maps after loading because it may take a lot of time for calibrations from large images on android
    if (!_calibration.load(_calibDir, Utils::getFileName(slamParams.calibrationFile), false))
    {
        showErrorMsg("Error when loading calibration from file: " +
                     slamParams.calibrationFile);
        return;
    }

    if (_calibration.imageSize() != _videoFrameSize)
    {
        _calibration.adaptForNewResolution(_videoFrameSize, true);
    }
    else
        _calibration.buildUndistortionMaps();

    // 3. Adjust FOV of camera node according to new calibration (fov is used in projection->prespective _mode)
    _s.updateCameraIntrinsics(_calibration.cameraFovVDeg(), _calibration.cameraMatUndistorted());
    //  _s.cameraNode->fov(_calibration.cameraFovVDeg());
    //// Set camera intrinsics for scene camera frustum. (used in projection->intrinsics mode)
    //cv::Mat scMat = _calibration.cameraMatUndistorted();
    //std::cout << "scMat: " << scMat << std::endl;
    //_s.cameraNode->intrinsics(scMat.at<double>(0, 0),
    //                                  scMat.at<double>(1, 1),
    //                                  scMat.at<double>(0, 2),
    //                                  scMat.at<double>(1, 2));

    ////enable projection -> intrinsics mode
    //_s.cameraNode->projection(P_monoIntrinsic);

    // 4. Create new mode ORBSlam
    if (!slamParams.markerFile.empty())
    {
        slamParams.params.cullRedundantPerc = 0.99f;
    }

    _trackingExtractor = _featureExtractorFactory.make(slamParams.extractorIds.trackingExtractorId, _videoFrameSize);
    /*
    _initializationExtractor = _featureExtractorFactory.make(slamParams->extractorIds.initializationExtractorId, _videoFrameSize);
    _markerExtractor         = _featureExtractorFactory.make(slamParams->extractorIds.markerExtractorId, _videoFrameSize);
    */
    _doubleBufferedOutput = _trackingExtractor->doubleBufferedOutput();

    ORBVocabulary* voc = new ORB_SLAM2::ORBVocabulary();
    voc->loadFromBinaryFile(slamParams.vocabularyFile);
    std::cout << "Vocabulary " << voc << std::endl;
    std::cout << "vocabulary file : " << slamParams.vocabularyFile << std::endl;
    WAIMap* map = nullptr;

    // 5. Load map data
    if (useMapFile)
    {
        std::cout << "Vocabulary " << voc << std::endl;
        WAIKeyFrameDB* kfdb    = new WAIKeyFrameDB(*voc);
        map                    = new WAIMap(kfdb);
        bool mapLoadingSuccess = WAIMapStorage::loadMap(map,
                                                        _s.mapNode,
                                                        voc,
                                                        slamParams.mapFile,
                                                        false, //TODO(lulu) add this param to slamParams _mode->retainImage(),
                                                        slamParams.params.fixOldKfs);

        if (!mapLoadingSuccess)
        {
            showErrorMsg("Could not load map from file " + slamParams.mapFile);
            return;
        }

        SlamMapInfos slamMapInfos = {};
        extractSlamMapInfosFromFileName(slamParams.mapFile, &slamMapInfos);
    }

    _mode = new WAISlam(_calibration.cameraMat(),
                        _calibration.distortion(),
                        voc,
                        _trackingExtractor.get(),
                        map,
                        slamParams.params.onlyTracking,
                        slamParams.params.serial,
                        slamParams.params.retainImg,
                        slamParams.params.cullRedundantPerc);

    // 6. save current params
    _currentSlamParams = slamParams;

    _sv.setViewportFromRatio(SLVec2i(_videoFrameSize.width, _videoFrameSize.height), SLViewportAlign::VA_center, true);
    //_resizeWindow = true;
    _undistortedLastFrame[0] = cv::Mat(_videoFrameSize.height, _videoFrameSize.width, CV_8UC3);
    _undistortedLastFrame[1] = cv::Mat(_videoFrameSize.height, _videoFrameSize.width, CV_8UC3);
}

//todo: move to scene
void TestState::transformMapNode(SLTransformSpace tSpace,
                                 SLVec3f          rotation,
                                 SLVec3f          translation,
                                 float            scale)
{
    _s.mapNode->rotate(rotation.x, 1, 0, 0, tSpace);
    _s.mapNode->rotate(rotation.y, 0, 1, 0, tSpace);
    _s.mapNode->rotate(rotation.z, 0, 0, 1, tSpace);
    _s.mapNode->translate(translation.x, 0, 0, tSpace);
    _s.mapNode->translate(0, translation.y, 0, tSpace);
    _s.mapNode->translate(0, 0, translation.z, tSpace);
    _s.mapNode->scale(scale);
}

void TestState::downloadCalibrationFilesTo(std::string dir)
{
    const std::string ftpHost = "pallas.bfh.ch:21";
    const std::string ftpUser = "upload";
    const std::string ftpPwd  = "FaAdbD3F2a";
    const std::string ftpDir  = "erleb-AR/calibrations/";
    std::string       errorMsg;
    if (!FtpUtils::downloadAllFilesFromDir(dir,
                                           ftpHost,
                                           ftpUser,
                                           ftpPwd,
                                           ftpDir,
                                           errorMsg))
    {
        errorMsg = "Failed to download calibration files. Error: " + errorMsg;
        this->showErrorMsg(errorMsg);
    }
}

void TestState::updateVideoTracking()
{
    while (_videoCursorMoveIndex < 0)
    {
        SENSFramePtr frame = _videoFileStream->grabPreviousFrame();
        if (frame)
        {
            if (_videoWriter && _videoWriter->isOpened())
                _videoWriter->write(frame->imgRGB);
            if (_mode)
                _mode->update(frame->imgGray);
        }

        _videoCursorMoveIndex++;
    }

    while (_videoCursorMoveIndex > 0)
    {
        SENSFramePtr frame = _videoFileStream->grabNextFrame();
        if (frame)
        {
            if (_videoWriter && _videoWriter->isOpened())
                _videoWriter->write(frame->imgRGB);
            if (_mode)
                _mode->update(frame->imgGray);
        }

        _videoCursorMoveIndex--;
    }
}

void TestState::updateTrackingVisualization(const bool iKnowWhereIAm, cv::Mat& imgRGB)
{
    //undistort image and copy image to video texture
    _mode->drawInfo(imgRGB, true, _gui.uiPrefs->showKeyPoints, _gui.uiPrefs->showKeyPointsMatched);

    if (_calibration.state() == CS_calibrated && _showUndistorted)
        _calibration.remap(imgRGB, _undistortedLastFrame[_lastFrameIdx]);
    else
        _undistortedLastFrame[_lastFrameIdx] = imgRGB;

    if (_doubleBufferedOutput)
        _lastFrameIdx = (_lastFrameIdx + 1) % 2;

    _s.updateVideoImage(_undistortedLastFrame[_lastFrameIdx],
                        CVImage::cv2glPixelFormat(_undistortedLastFrame[_lastFrameIdx].type()));

    //update map point visualization
    if (_gui.uiPrefs->showMapPC)
    {
        _s.renderMapPoints(_mode->getMapPoints());
        //todo: fix? mode has no member getMarmerCornerMapPoints
        //_s.renderMarkerCornerMapPoints(_mode->getMarkerCornerMapPoints());
    }
    else
    {
        _s.removeMapPoints();
        _s.removeMarkerCornerMapPoints();
    }

    //update visualization of local map points (when WAI pose is valid)
    if (_gui.uiPrefs->showLocalMapPC && iKnowWhereIAm)
        _s.renderLocalMapPoints(_mode->getLocalMapPoints());
    else
        _s.removeLocalMapPoints();

    //update visualization of matched map points (when WAI pose is valid)
    if (_gui.uiPrefs->showMatchesPC && iKnowWhereIAm)
        _s.renderMatchedMapPoints(_mode->getMatchedMapPoints(_mode->getLastFrame()));
    else
        _s.removeMatchedMapPoints();

    //update keyframe visualization
    if (_gui.uiPrefs->showKeyFrames)
        _s.renderKeyframes(_mode->getKeyFrames());
    else
        _s.removeKeyframes();

    //update pose graph visualization
    _s.renderGraphs(_mode->getKeyFrames(),
                    _gui.uiPrefs->minNumOfCovisibles,
                    _gui.uiPrefs->showCovisibilityGraph,
                    _gui.uiPrefs->showSpanningTree,
                    _gui.uiPrefs->showLoopEdges);
}

void TestState::setupDefaultErlebARDirTo(std::string dir)
{
    dir = Utils::unifySlashes(dir);
    if (!Utils::dirExists(dir))
    {
        Utils::makeDir(dir);
    }
    //calibrations directory
    if (!Utils::dirExists(dir + "calibrations/"))
    {
        Utils::makeDir(dir + "calibrations/");
    }

    dir += "erleb-AR/";
    if (!Utils::dirExists(dir))
    {
        Utils::makeDir(dir);
    }

    dir += "locations/";
    if (!Utils::dirExists(dir))
    {
        Utils::makeDir(dir);
    }

    dir += "default/";
    if (!Utils::dirExists(dir))
    {
        Utils::makeDir(dir);
    }

    dir += "default/";
    if (!Utils::dirExists(dir))
    {
        Utils::makeDir(dir);
    }
}
