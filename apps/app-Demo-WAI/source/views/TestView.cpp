#include <views/TestView.h>
#include <WAISlam.h>
#include <WAIEvent.h>
#include <sens/SENSCamera.h>
#include <WAIMapStorage.h>
#include <AppWAISlamParamHelper.h>
#include <FtpUtils.h>
#include <WAIAutoCalibration.h>
#include <sens/SENSUtils.h>

#define LOG_TESTVIEW_WARN(...) Utils::log("TestView", __VA_ARGS__);
#define LOG_TESTVIEW_INFO(...) Utils::log("TestView", __VA_ARGS__);
#define LOG_TESTVIEW_DEBUG(...) Utils::log("TestView", __VA_ARGS__);

TestView::TestView(sm::EventHandler&   eventHandler,
                   SLInputManager&     inputManager,
                   const ImGuiEngine&  imGuiEngine,
                   ErlebAR::Resources& resources,
                   SENSCamera*         camera,
                   const DeviceData&   deviceData)
  : SLSceneView(&_scene, deviceData.dpi(), inputManager),
    _gui(
      imGuiEngine,
      eventHandler,
      resources,
      "TestScene",
      deviceData,
      _featureExtractorFactory.getExtractorIdToNames(),
      _eventQueue,
      [&]() { return _mode; },                            //getter callback for current mode
      [&]() { return _camera; },                          //getter callback for current camera
      [&]() { return _camera ? _calibration : nullptr; }, //getter callback for current calibration
      [&]() { return _videoFileStream.get(); }),          //getter callback for current calibration
    _scene("TestScene", deviceData.dataDir()),
    _camera(camera),
    _configDir(deviceData.writableDir()),
    _vocabularyDir(deviceData.vocabularyDir()),
    _calibDir(deviceData.erlebARCalibTestDir()),
    _videoDir(deviceData.erlebARTestDir() + "videos/"),
    _dataDir(deviceData.dataDir())
{
    scene(&_scene);
    init("TestSceneView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, &_gui, _configDir);
    _scene.init();
    onInitialize();
    _fillAutoCalibration = false;
    _voc                 = new WAIOrbVocabulary();

    setupDefaultErlebARDirTo(deviceData.erlebARTestDir());
    //tryLoadLastSlam();
}

TestView::~TestView()
{
    if (_mode)
    {
        delete _mode;
        _mode = nullptr;
    }
}

void TestView::start()
{
    tryLoadLastSlam();
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
        frame = _camera->latestFrame();
    else
        Utils::log("WAI WARN", "TestView::update: No active camera or video stream available!");

    if (frame)
    {
        if (_videoWriter && _videoWriter->isOpened())
            _videoWriter->write(frame->imgRGB);

        if (_mode)
        {
            _mode->update(frame->imgManip);

            if (_mode->isTracking())
            {
                _scene.updateCameraPose(_mode->getPose());

                if (_fillAutoCalibration)
                {
                    WAIFrame                                                      lastFrame = _mode->getLastFrame();
                    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f>> matching;
                    _mode->getMatchedCorrespondances(&lastFrame, matching);
                    _autoCal->fillFrame(matching, lastFrame.mTcw);
                }
            }

            if (_autoCal && _autoCal->hasCalibration())
            {
                _camera->setCalibration(_autoCal->consumeCalibration(), true);
                _calibration = _camera->calibration();
                _scene.updateCameraIntrinsics(_calibration->cameraFovVDeg());
                cv::Mat scaledCamMat = SENS::adaptCameraMat(_calibration->cameraMat(),
                                                            _camera->config().manipWidth,
                                                            _camera->config().targetWidth);
                _mode->changeIntrinsic(scaledCamMat, _calibration->distortion());
                _fillAutoCalibration = false;
            }
            updateTrackingVisualization(_mode->isTracking(), *frame.get());
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

            case WAIEventType_AutoCalibration: {
                WAIEventAutoCalibration* autoCalEvent = (WAIEventAutoCalibration*)event;
                if (autoCalEvent->tryCalibrate)
                {
                    _autoCal->reset();
                    _fillAutoCalibration = true;
                }
                else if (autoCalEvent->useGuessCalibration)
                {
                    auto  streamConfig = _camera->config().streamConfig;
                    float horizFovDeg  = 65.f;
                    if (streamConfig.focalLengthPix > 0)
                        horizFovDeg = SENS::calcFOVDegFromFocalLengthPix(streamConfig.focalLengthPix, streamConfig.widthPix);

                    //_calibration = CVCalibration(_videoFrameSize, horizFOVDev, false, false, CVCameraType::BACKFACING, Utils::ComputerInfos::get());
                    SENSCalibration calib(cv::Size(streamConfig.widthPix, streamConfig.heightPix), horizFovDeg, false, false, SENSCameraType::BACKFACING, Utils::ComputerInfos::get());
                    _camera->setCalibration(calib, false);
                    _calibration = _camera->calibration();
                    _scene.updateCameraIntrinsics(_calibration->cameraFovVDeg());
                    cv::Mat scaledCamMat = SENS::adaptCameraMat(_calibration->cameraMat(),
                                                                _camera->config().manipWidth,
                                                                _camera->config().targetWidth);
                    _mode->changeIntrinsic(scaledCamMat, _calibration->distortion());
                }
                else if (autoCalEvent->restoreOriginalCalibration)
                {
                    _camera->setCalibration(*_calibrationLoaded, true);
                    _calibration = _camera->calibration();
                    _scene.updateCameraIntrinsics(_calibration->cameraFovVDeg());
                    cv::Mat scaledCamMat = SENS::adaptCameraMat(_calibration->cameraMat(),
                                                                _camera->config().manipWidth,
                                                                _camera->config().targetWidth);
                    _mode->changeIntrinsic(scaledCamMat, _calibration->distortion());
                }
                delete autoCalEvent;
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

                transformMapNode(mapNodeTransformEvent->tSpace,
                                 mapNodeTransformEvent->rotation,
                                 mapNodeTransformEvent->translation,
                                 mapNodeTransformEvent->scale);

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
            case WAIEventType_EditMap: {
                WAIEventEditMap* editMap = (WAIEventEditMap*)event;

                if (editMap->action == MapPointEditor_EnterEditMode && !_mapEdition)
                {
                    _scene.removeLocalMapPoints();
                    _scene.removeMapPoints();
                    _scene.removeKeyframes();
                    _scene.removeMatchedMapPoints();
                    _scene.removeMarkerCornerMapPoints();
                    _scene.removeGraphs();
                    _mapEdition = new MapEdition(this, _scene.root3D()->findChild<SLNode>("map"), _mode->getMap(), _dataDir + "shaders/");
                    _scene.root3D()->addChild(_mapEdition);
                    _pauseVideo = true;
                }
                else if (editMap->action == MapPointEditor_ApplyToMapPoints && _mapEdition)
                {
                    SLNode*      mapNode = _scene.root3D()->findChild<SLNode>("map");
                    const float* m       = mapNode->om().m();
                    cv::Mat      mat     = (cv::Mat_<float>(4, 4) << m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14], m[3], m[7], m[11], m[15]);
                    _mode->transformCoords(mat);
                    _scene.resetMapNode();
                    _mapEdition->updateVisualization();
                }
                else if (editMap->action == MapPointEditor_SaveMap && _mapEdition)
                    saveMap(_currentSlamParams.location, _currentSlamParams.area, _currentSlamParams.markerFile);
                else if (editMap->action == MapPointEditor_SaveMapRaw && _mapEdition)
                    saveMap(_currentSlamParams.location, _currentSlamParams.area, _currentSlamParams.markerFile, true);
                else if (editMap->action == MapPointEditor_LoadMatching && _mapEdition)
                    _mapEdition->updateKFVidMatching(editMap->kFVidMatching);
                else if (editMap->action == MapPointEditor_SelectSingleVideo && _mapEdition)
                    _mapEdition->selectByVid(editMap->vid);
                else if (editMap->action == MapPointEditor_SelectNMatched && _mapEdition)
                    _mapEdition->selectNMatched(editMap->nmatches);
                else if (editMap->action == MapPointEditor_KeyFrameMode && _mapEdition)
                    _mapEdition->setKeyframeMode(editMap->b);
                else if (editMap->action == MapPointEditor_SelectAllPoints && _mapEdition)
                    _mapEdition->selectAllMap();
                else if (editMap->action == MapPointEditor_Quit && _mapEdition)
                {
                    if (_scene.root3D()->deleteChild(_mapEdition))
                    {
                        auto it = find(_scene.eventHandlers().begin(),
                                       _scene.eventHandlers().end(),
                                       _mapEdition);
                        if (it != _scene.eventHandlers().end())
                            _scene.eventHandlers().erase(it);

                        _mapEdition = nullptr;
                    }
                    updateTrackingVisualization(_mode);
                }
                else if (_mapEdition)
                {
                    _mapEdition->selectAllMap();
                    _mapEdition->editMode(editMap->editMode);
                }
                delete editMap;
            }
            break;

            case WAIEventType_None:
            default: {
            }
            break;
        }
    }
}

//void TestView::postStart()
//{
//    doWaitOnIdle(false);
//    camera(_scene.cameraNode);
//    onInitialize();
//    if (_camera)
//        setViewportFromRatio(SLVec2i(_camera->config().targetWidth, _camera->config().targetHeight), SLViewportAlign::VA_center, true);
//}

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
                       std::string marker,
                       bool raw)
{
    _mode->requestStateIdle();

    std::string slamRootDir = _configDir + "erleb-AR/locations/";
    std::string mapDir      = constructSlamMapDir(slamRootDir, location, area);
    if (!Utils::dirExists(mapDir))
        Utils::makeDir(mapDir);

    std::string filename = constructSlamMapFileName(location, area, _mode->getKPextractor()->GetName(), _mode->getKPextractor()->GetLevels());
    std::string imgDir   = constructSlamMapImgDir(mapDir, filename);

    if (_mode->retainImage())
    {
        if (!Utils::dirExists(imgDir))
            Utils::makeDir(imgDir);
    }

    if (!marker.empty())
    {
        _voc->loadFromFile(_currentSlamParams.vocabularyFile);

        cv::Mat nodeTransform;
        if (!WAISlamTools::doMarkerMapPreprocessing(constructSlamMarkerDir(slamRootDir, location, area) + marker,
                                                    nodeTransform,
                                                    0.75f,
                                                    _mode->getKPextractor(),
                                                    _mode->getMap(),
                                                    _calibration->cameraMat(),
                                                    _voc))
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
        if (raw)
        {
            if (!WAIMapStorage::saveMapRaw(_mode->getMap(),
                                           _scene.mapNode,
                                           mapDir + filename,
                                           imgDir))
            {
                _gui.showErrorMsg("Failed to save map " + mapDir + filename);
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

void TestView::updateSceneCameraFov()
{
    //if the camera image height is smaller than the sceneview height,
    //we have to calculate the corresponding vertical field of view for the scene camera
    float imgWdivH = _calibration->imageAspectRatio();
    if (std::abs(this->scrWdivH() - imgWdivH) > 0.00001)
    {
        if (this->scrWdivH() > imgWdivH)
        {
            //bars left and right: directly use camera vertial field of view as scene vertical field of view
            _scene.updateCameraIntrinsics(_calibration->cameraFovVDeg());
        }
        else
        {
            //bars top and bottom: estimate vertical fov from cameras horizontal field of view and screen aspect ratio
            float fovV = SENS::calcFovDegFromOtherFovDeg(_calibration->cameraFovHDeg(), this->scrW(), this->scrH());
            _scene.updateCameraIntrinsics(fovV);
        }
    }
    else
    {
        //no bars because same aspect ration: directly use camera vertial field of view as scene vertical field of view
        _scene.updateCameraIntrinsics(_calibration->cameraFovVDeg());
    }
}

bool TestView::startCamera()
{
    if (!_camera)
    {
        _gui.showErrorMsg("Camera pointer is not set!");
        return false;
    }

    if (_camera->started())
        _camera->stop();

    auto capProps     = _camera->captureProperties();
    int  trackingImgW = 640;

    float targetWdivH   = 4.f / 3.f;
    int   aproxVisuImgW = 640;
    int   aproxVisuImgH = (int)((float)aproxVisuImgW / targetWdivH);

    auto bestConfig = capProps.findBestMatchingConfig(SENSCameraFacing::BACK, 65.f, aproxVisuImgW, aproxVisuImgH);

    if (bestConfig.first && bestConfig.second)
    {
        const SENSCameraDeviceProperties* const devProps     = bestConfig.first;
        const SENSCameraStreamConfig*           streamConfig = bestConfig.second;
        //calculate size of tracking image
        float imgWdivH = (float)streamConfig->widthPix / (float)streamConfig->heightPix;
        _camera->start(devProps->deviceId(),
                       *streamConfig,
                       cv::Size(),
                       false,
                       false,
                       true,
                       trackingImgW,
                       true,
                       65.f);
    }

    else //try with unknown config (for desktop)
    {
        aproxVisuImgW    = 640;
        auto bestConfig2 = capProps.findBestMatchingConfig(SENSCameraFacing::UNKNOWN, 65.f, 640, 480);
        if (bestConfig2.first && bestConfig2.second)
        {
            const SENSCameraDeviceProperties* const devProps     = bestConfig2.first;
            const SENSCameraStreamConfig*           streamConfig = bestConfig2.second;
            //calculate size of tracking image
            float imgWdivH = (float)streamConfig->widthPix / (float)streamConfig->heightPix;
            _camera->start(devProps->deviceId(),
                           *streamConfig,
                           cv::Size(),
                           false,
                           false,
                           true,
                           trackingImgW,
                           true,
                           65.f);
        }
    }

    return _camera->started();
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

    if (_autoCal)
    {
        delete _autoCal;
        _autoCal = nullptr;
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
        if (!startCamera())
            return;
        _videoFrameSize = cv::Size2i(_camera->config().targetWidth, _camera->config().targetHeight);
    }

    // 2. Load Calibration
    //build undistortion maps after loading because it may take a lot of time for calibrations from large images on android
    try
    {
        _calibrationLoaded = std::make_unique<SENSCalibration>(_calibDir, Utils::getFileName(slamParams.calibrationFile), false);
    }
    catch (SENSException& e)
    {
        _gui.showErrorMsg("Error when loading calibration from file: " +
                          slamParams.calibrationFile);
        return;
    }

    //This really does not look nice.. That I do it like this is because the iOS camera provides camera intrinsics directly for every frame
    //and android also provides constant intrinsics (by focal length). When I finished it I realized that it does not really fit to the SENSVideoStream concept.
    //Maybe we find something better..?
    if (useVideoFile)
    {
        _videoFileStream->setCalibration(*_calibrationLoaded, true);
        _calibration = _videoFileStream->calibration();
    }
    else
    {
        //set and adapt calibration to camera image resolution
        _camera->setCalibration(*_calibrationLoaded, true);
        _calibration = _camera->calibration();
    }

    // 3. Adjust FOV of camera node according to new calibration (fov is used in projection->prespective _mode)
    _scene.updateCameraIntrinsics(_calibration->cameraFovVDeg());

    // 4. Create new mode ORBSlam
    if (!slamParams.markerFile.empty())
    {
        slamParams.params.cullRedundantPerc = 0.99f;
    }

    _trackingExtractor       = _featureExtractorFactory.make(slamParams.extractorIds.trackingExtractorId, _videoFrameSize, slamParams.nLevels);
    _initializationExtractor = _featureExtractorFactory.make(slamParams.extractorIds.initializationExtractorId, _videoFrameSize, slamParams.nLevels);

    try
    {
        _voc->loadFromFile(slamParams.vocabularyFile);
    }
    catch (std::exception& e)
    {
        _gui.showErrorMsg("Could not load vocabulary file from " + slamParams.vocabularyFile);
        return;
    }
    std::cout << "vocabulary file : " << slamParams.vocabularyFile << std::endl;
    std::unique_ptr<WAIMap> map;

    // 5. Load map data
    if (useMapFile)
    {
        WAIKeyFrameDB* kfdb    = new WAIKeyFrameDB(_voc);
        map                    = std::make_unique<WAIMap>(kfdb);
        bool mapLoadingSuccess = WAIMapStorage::loadMap(map.get(),
                                                        _scene.mapNode,
                                                        _voc,
                                                        slamParams.mapFile,
                                                        false, //TODO(lulu) add this param to slamParams _mode->retainImage(),
                                                        slamParams.params.fixOldKfs);

        _autoCal = new AutoCalibration(_videoFrameSize, map->GetSize());

        if (!mapLoadingSuccess)
        {
            _gui.showErrorMsg("Could not load map from file " + slamParams.mapFile);
            return;
        }

        SlamMapInfos slamMapInfos = {};
        extractSlamMapInfosFromFileName(slamParams.mapFile, &slamMapInfos);
    }

    //I KNOW THIS IS STILL ALL SHIT: THIS IS A HOTFIX BEFORE HOLIDAY
    cv::Mat scaledCamMat;
    if (useVideoFile)
    {
        scaledCamMat = _calibration->cameraMat();
    }
    else
    {
        scaledCamMat = SENS::adaptCameraMat(_calibration->cameraMat(),
                                            _camera->config().manipWidth,
                                            _camera->config().targetWidth);
    }
    _mode = new WAISlam(scaledCamMat,
                        _calibration->distortion(),
                        _voc,
                        _initializationExtractor.get(),
                        _trackingExtractor.get(),
                        std::move(map),
                        slamParams.params.onlyTracking,
                        slamParams.params.serial,
                        slamParams.params.retainImg,
                        slamParams.params.cullRedundantPerc);

    // 6. save current params
    _currentSlamParams = slamParams;
    _gui.setSlamParams(slamParams);
    _currentSlamParams.save(_configDir + "SlamParams.json");

    setViewportFromRatio(SLVec2i(_videoFrameSize.width, _videoFrameSize.height), SLViewportAlign::VA_center, true);

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
    const std::string ftpDir  = "erleb-AR-data/test/erleb-AR/calibrations/";
    std::string       errorMsg;
    if (!FtpUtils::downloadAllFilesFromDir(dir,
                                           ftpHost,
                                           ftpUser,
                                           ftpPwd,
                                           ftpDir,
                                           "xml",
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
                _mode->update(frame->imgManip);
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
                _mode->update(frame->imgManip);
        }

        _videoCursorMoveIndex--;
    }
}

void TestView::updateTrackingVisualization(const bool iKnowWhereIAm)
{
    if (!_mode)
        return;

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

void TestView::updateTrackingVisualization(const bool iKnowWhereIAm, SENSFrame& frame)
{
    //undistort image and copy image to video texture
    _mode->drawInfo(frame.imgRGB, frame.scaleToManip, true, _gui.uiPrefs->showKeyPoints, _gui.uiPrefs->showKeyPointsMatched);

    if (_calibration->state() == SENSCalibration::State::calibrated && _showUndistorted)
        _calibration->remap(frame.imgRGB, _imgBuffer.inputSlot());
    else
        _imgBuffer.inputSlot() = frame.imgRGB;

    _scene.updateVideoImage(_imgBuffer.outputSlot());
    _imgBuffer.incrementSlot();

    updateTrackingVisualization(iKnowWhereIAm);
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
