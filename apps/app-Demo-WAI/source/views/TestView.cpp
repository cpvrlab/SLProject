#include <views/TestView.h>
#include <WAISlam.h>
#include <WAIEvent.h>
#include <sens/SENSCamera.h>
#include <WAIMapStorage.h>
#include <AppWAISlamParamHelper.h>
#include <FtpUtils.h>
#include <WAIAutoCalibration.h>

#define LOG_TESTVIEW_WARN(...) Utils::log("TestView", __VA_ARGS__);
#define LOG_TESTVIEW_INFO(...) Utils::log("TestView", __VA_ARGS__);
#define LOG_TESTVIEW_DEBUG(...) Utils::log("TestView", __VA_ARGS__);

TestView::TestView(sm::EventHandler& eventHandler,
                   SLInputManager&   inputManager,
                   SENSCamera*       camera,
                   int               screenWidth,
                   int               screenHeight,
                   int               dotsPerInch,
                   std::string       fontPath,
                   std::string       configDir,
                   std::string       vocabularyDir,
                   std::string       calibDir,
                   std::string       videoDir)
  : SLSceneView(&_scene, dotsPerInch, inputManager),
    _gui(
      eventHandler,
      "TestScene",
      dotsPerInch,
      screenWidth,
      screenHeight,
      configDir,
      fontPath,
      vocabularyDir,
      _featureExtractorFactory.getExtractorIdToNames(),
      _eventQueue,
      [&]() { return _mode; },                   //getter callback for current mode
      [&]() { return _camera; },                 //getter callback for current camera
      [&]() { return &_calibration; },           //getter callback for current calibration
      [&]() { return _videoFileStream.get(); }), //getter callback for current calibration
    _scene("TestScene"),
    _camera(camera),
    _configDir(configDir),
    _vocabularyDir(vocabularyDir),
    _calibDir(calibDir),
    _videoDir(videoDir)
{
    scene(&_scene);
    init("TestSceneView", screenWidth, screenHeight, nullptr, nullptr, &_gui, _configDir);
    _scene.init();
    onInitialize();
    _isCalibrated = false;

    setupDefaultErlebARDirTo(_configDir);
    //tryLoadLastSlam();
}

TestView::~TestView()
{
    if (_mode)
    {
        delete _mode;
        _mode = nullptr;
        _currentSlamParams.save(_configDir + "SlamParams.json");
    }

    if (_startThread.joinable())
    {
        _startThread.join();
    }
}

void TestView::start()
{
    //if (_ready)
    //    return;

    //_startThread = std::thread(&TestView::startAsync, this);
    tryLoadLastSlam();
}

void TestView::startAsync()
{
    //_camera->init(SENSCameraFacing::BACK);
    ////start continious captureing request with certain configuration
    //SENSCameraConfig camConfig;
    //camConfig.targetWidth          = 640;
    //camConfig.targetHeight         = 360;
    //camConfig.focusMode            = SENSCamera::SENSCameraFocusMode::FIXED_INFINITY_FOCUS;
    //camConfig.convertToGray        = true;
    //camConfig.adjustAsynchronously = true;
    //_camera->start(camConfig);

    //start thread that starts camera and tries to load slam
    tryLoadLastSlam();

    //_ready = true;
}

bool TestView::update()
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
        Utils::log("WAI WARN", "TestView::update: No active camera or video stream available!");

    if (frame)
    {
        if (_videoWriter && _videoWriter->isOpened())
            _videoWriter->write(frame->imgRGB);

        if (_mode)
        {
            _mode->update(frame->imgGray);

            if (_mode->isTracking())
            {
                _scene.updateCameraPose(_mode->getPose());

                if (!_isCalibrated)
                {
                    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>> matching;

                    WAIFrame lastFrame = _mode->getLastFrame();
                    if (_mode->getMatchedCorrespondances(&lastFrame, matching) > 10)
                    {
                        _calibrationMatchings.push_back(matching);
                        if (_calibrationMatchings.size() > 10)
                        {
                            cv::Size size      = cv::Size(frame->captureWidth, frame->captureHeight);
                            _calibrationThread = std::thread(AutoCalibration::calibrate, size, _calibrationMatchings);
                            _calibrationMatchings.clear();
                            _isCalibrated = true;
                        }
                    }
                }
            }

            updateTrackingVisualization(_mode->isTracking(), frame->imgRGB);
        }
    }

    return onPaint();
}

void TestView::tryLoadLastSlam()
{
    if (_currentSlamParams.load(_configDir + "SlamParams.json"))
    {
        //_scene.rebuild(_currentSlamParams.location, _currentSlamParams.area);
        loadWAISceneView(_currentSlamParams.location, _currentSlamParams.area);
        startOrbSlam(_currentSlamParams);
        //_guiSlamLoad->setSlamParams(_currentSlamParams);
        _gui.setSlamParams(_currentSlamParams);
        _gui.uiPrefs->showSlamLoad = false;
    }
    else
    {
        _gui.uiPrefs->showSlamLoad = true;
    }
}

void TestView::handleEvents()
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
                _scene.adjustAugmentationTransparency(adjustTransparencyEvent->kt);

                delete adjustTransparencyEvent;
            }
            break;

            case WAIEventType_EnterEditMode: {
                WAIEventEnterEditMode* enterEditModeEvent = (WAIEventEnterEditMode*)event;

                if (!_transformationNode)
                {
                    _transformationNode = new SLTransformNode(&_assets, this, _scene.root3D()->findChild<SLNode>("map"));
                    _scene.root3D()->addChild(_transformationNode);
                }

                if (enterEditModeEvent->editMode == NodeEditMode_None)
                {
                    if (_scene.root3D()->deleteChild(_transformationNode))
                    {
                        _transformationNode = nullptr;
                    }
                }
                else
                {
                    _transformationNode->editMode(enterEditModeEvent->editMode);
                }

                delete enterEditModeEvent;
            }
            break;

            case WAIEventType_None:
            default: {
            }
            break;
        }
    }
}

void TestView::postStart()
{
    doWaitOnIdle(false);
    camera(_scene.cameraNode);
    onInitialize();
    if (_camera)
        setViewportFromRatio(SLVec2i(_camera->config().targetWidth, _camera->config().targetHeight), SLViewportAlign::VA_center, true);
}

void TestView::loadWAISceneView(std::string location, std::string area)
{
    _scene.rebuild(location, area);

    doWaitOnIdle(false);
    camera(_scene.cameraNode);
    onInitialize();
    if (_camera)
        setViewportFromRatio(SLVec2i(_camera->config().targetWidth, _camera->config().targetHeight), SLViewportAlign::VA_center, true);
}

void TestView::saveMap(std::string location,
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
            _gui.showErrorMsg("Failed to do marker map preprocessing");
        }
        else
        {
            std::cout << "nodeTransform: " << nodeTransform << std::endl;
            //_scene.mapNode->om(WAIMapStorage::convertToSLMat(nodeTransform));
            if (!WAIMapStorage::saveMap(_mode->getMap(),
                                        _scene.mapNode,
                                        mapDir + filename,
                                        imgDir))
            {
                _gui.showErrorMsg("Failed to save map " + mapDir + filename);
            }
        }
    }
    else
    {
        if (!WAIMapStorage::saveMap(_mode->getMap(),
                                    _scene.mapNode,
                                    mapDir + filename,
                                    imgDir))
        {
            _gui.showErrorMsg("Failed to save map " + mapDir + filename);
        }
    }

    _mode->resume();
}

void TestView::saveVideo(std::string filename)
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
        ret = _videoWriter->open(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(_camera->config().targetWidth, _camera->config().targetHeight), true);
    else
        Utils::log("WAI WARN", "WAIApp::saveVideo: No active video stream or camera available!");
}

/*
videoFile: path to a video or empty if live video should be used
calibrationFile: path to a calibration or empty if calibration should be searched automatically
mapFile: path to a map or empty if no map should be used
*/
void TestView::startOrbSlam(SlamParams slamParams)
{
    _gui.clearErrorMsg();
    if (_videoFileStream)
        _videoFileStream.release();

    bool useVideoFile             = !slamParams.videoFile.empty();
    bool detectCalibAutomatically = slamParams.calibrationFile.empty();
    bool useMapFile               = !slamParams.mapFile.empty();

    // reset stuff
    if (_mode)
    {
        delete _mode;
        _mode = nullptr;
    }

    // Check that files exist
    if (useVideoFile && !Utils::fileExists(slamParams.videoFile))
    {
        _gui.showErrorMsg("Video file " + slamParams.videoFile + " does not exist.");
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
                _gui.showErrorMsg("Could not extract computer infos from video filename.");
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
        _gui.showErrorMsg("Calibration file " + slamParams.calibrationFile + " does not exist.");
        return;
    }

    /*
    if (!checkCalibration(calibDir, calibrationFileName))
    {
        _gui.showErrorMsg("Calibration file " + calibrationFile + " is incorrect.");
        return;
    }
     */

    if (slamParams.vocabularyFile.empty())
    {
        _gui.showErrorMsg("Select a vocabulary file!");
        return;
    }

    if (!Utils::fileExists(slamParams.vocabularyFile))
    {
        _gui.showErrorMsg("Vocabulary file does not exist: " + slamParams.vocabularyFile);
        return;
    }

    if (useMapFile && !Utils::fileExists(slamParams.mapFile))
    {
        _gui.showErrorMsg("Map file " + slamParams.mapFile + " does not exist.");
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
            _gui.showErrorMsg("Camera pointer is not set!");
            return;
        }
        _videoFrameSize = cv::Size2i(_camera->config().targetWidth, _camera->config().targetHeight);
    }

    // 2. Load Calibration
    //build undistortion maps after loading because it may take a lot of time for calibrations from large images on android
    if (!_calibration.load(_calibDir, Utils::getFileName(slamParams.calibrationFile), false))
    {
        _gui.showErrorMsg("Error when loading calibration from file: " +
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
    _scene.updateCameraIntrinsics(_calibration.cameraFovVDeg(), _calibration.cameraMatUndistorted());
    //  _scene.cameraNode->fov(_calibration.cameraFovVDeg());
    //// Set camera intrinsics for scene camera frustum. (used in projection->intrinsics mode)
    //cv::Mat scMat = _calibration.cameraMatUndistorted();
    //std::cout << "scMat: " << scMat << std::endl;
    //_scene.cameraNode->intrinsics(scMat.at<double>(0, 0),
    //                                  scMat.at<double>(1, 1),
    //                                  scMat.at<double>(0, 2),
    //                                  scMat.at<double>(1, 2));

    ////enable projection -> intrinsics mode
    //_scene.cameraNode->projection(P_monoIntrinsic);

    // 4. Create new mode ORBSlam
    if (!slamParams.markerFile.empty())
    {
        slamParams.params.cullRedundantPerc = 0.99f;
    }

    _trackingExtractor       = _featureExtractorFactory.make(slamParams.extractorIds.trackingExtractorId, _videoFrameSize);
    _initializationExtractor = _featureExtractorFactory.make(slamParams.extractorIds.initializationExtractorId, _videoFrameSize);
    //_doubleBufferedOutput    = _trackingExtractor->doubleBufferedOutput();

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
                                                        _scene.mapNode,
                                                        voc,
                                                        slamParams.mapFile,
                                                        false, //TODO(lulu) add this param to slamParams _mode->retainImage(),
                                                        slamParams.params.fixOldKfs);

        if (!mapLoadingSuccess)
        {
            _gui.showErrorMsg("Could not load map from file " + slamParams.mapFile);
            return;
        }

        SlamMapInfos slamMapInfos = {};
        extractSlamMapInfosFromFileName(slamParams.mapFile, &slamMapInfos);
    }

    _mode = new WAISlam(_calibration.cameraMat(),
                        _calibration.distortion(),
                        voc,
                        _initializationExtractor.get(),
                        _trackingExtractor.get(),
                        map,
                        slamParams.params.onlyTracking,
                        slamParams.params.serial,
                        slamParams.params.retainImg,
                        slamParams.params.cullRedundantPerc);

    // 6. save current params
    _currentSlamParams = slamParams;

    setViewportFromRatio(SLVec2i(_videoFrameSize.width, _videoFrameSize.height), SLViewportAlign::VA_center, true);
    //_resizeWindow = true;

    if (_trackingExtractor->doubleBufferedOutput())
        _imgBuffer.init(2, _videoFrameSize);
    else
        _imgBuffer.init(1, _videoFrameSize);
}

//todo: move to scene
void TestView::transformMapNode(SLTransformSpace tSpace,
                                SLVec3f          rotation,
                                SLVec3f          translation,
                                float            scale)
{
    _scene.mapNode->rotate(rotation.x, 1, 0, 0, tSpace);
    _scene.mapNode->rotate(rotation.y, 0, 1, 0, tSpace);
    _scene.mapNode->rotate(rotation.z, 0, 0, 1, tSpace);
    _scene.mapNode->translate(translation.x, 0, 0, tSpace);
    _scene.mapNode->translate(0, translation.y, 0, tSpace);
    _scene.mapNode->translate(0, 0, translation.z, tSpace);
    _scene.mapNode->scale(scale);
}

void TestView::downloadCalibrationFilesTo(std::string dir)
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
        _gui.showErrorMsg(errorMsg);
    }
}

void TestView::updateVideoTracking()
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

void TestView::updateTrackingVisualization(const bool iKnowWhereIAm, cv::Mat& imgRGB)
{
    //undistort image and copy image to video texture
    _mode->drawInfo(imgRGB, true, _gui.uiPrefs->showKeyPoints, _gui.uiPrefs->showKeyPointsMatched);

    if (_calibration.state() == CS_calibrated && _showUndistorted)
        _calibration.remap(imgRGB, _imgBuffer.inputSlot());
    else
        _imgBuffer.inputSlot() = imgRGB;

    _scene.updateVideoImage(_imgBuffer.outputSlot());
    _imgBuffer.incrementSlot();

    //update map point visualization
    if (_gui.uiPrefs->showMapPC)
    {
        _scene.renderMapPoints(_mode->getMapPoints());
        //todo: fix? mode has no member getMarmerCornerMapPoints
        //_scene.renderMarkerCornerMapPoints(_mode->getMarkerCornerMapPoints());
    }
    else
    {
        _scene.removeMapPoints();
        _scene.removeMarkerCornerMapPoints();
    }

    //update visualization of local map points (when WAI pose is valid)
    if (_gui.uiPrefs->showLocalMapPC && iKnowWhereIAm)
        _scene.renderLocalMapPoints(_mode->getLocalMapPoints());
    else
        _scene.removeLocalMapPoints();

    //update visualization of matched map points (when WAI pose is valid)
    if (_gui.uiPrefs->showMatchesPC && iKnowWhereIAm)
        _scene.renderMatchedMapPoints(_mode->getMatchedMapPoints(_mode->getLastFramePtr()));
    else
        _scene.removeMatchedMapPoints();

    //update keyframe visualization
    if (_gui.uiPrefs->showKeyFrames)
        _scene.renderKeyframes(_mode->getKeyFrames());
    else
        _scene.removeKeyframes();

    //update pose graph visualization
    _scene.renderGraphs(_mode->getKeyFrames(),
                        _gui.uiPrefs->minNumOfCovisibles,
                        _gui.uiPrefs->showCovisibilityGraph,
                        _gui.uiPrefs->showSpanningTree,
                        _gui.uiPrefs->showLoopEdges);
}

void TestView::setupDefaultErlebARDirTo(std::string dir)
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
