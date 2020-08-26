#include <views/AreaTrackingView.h>
#include <sens/SENSCamera.h>
#include <WAIMapStorage.h>
#include <sens/SENSUtils.h>

AreaTrackingView::AreaTrackingView(sm::EventHandler&   eventHandler,
                                   SLInputManager&     inputManager,
                                   const ImGuiEngine&  imGuiEngine,
                                   ErlebAR::Resources& resources,
                                   SENSCamera*         camera,
                                   const DeviceData&   deviceData)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    _gui(imGuiEngine,
         eventHandler,
         resources,
         deviceData.dpi(),
         deviceData.scrWidth(),
         deviceData.scrHeight(),
         std::bind(&AppWAIScene::adjustAugmentationTransparency, &_scene, std::placeholders::_1)),
    _scene("AreaTrackingScene", deviceData.dataDir()),
    _camera(camera),
    _vocabularyDir(deviceData.vocabularyDir()),
    _erlebARDir(deviceData.erlebARDir())
{
    scene(&_scene);
    init("AreaTrackingView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, &_gui, deviceData.writableDir());
    //todo: ->moved to constructor of AreaTrackingScene: can this lead to any problems?
    //_scene.init();
    //_scene.build();
    onInitialize();
    _voc = new WAIOrbVocabulary();
    //set scene camera into sceneview
    //(camera node not initialized yet)
    //this->camera(_scene.cameraNode);

    _locations = resources.locations();
}

AreaTrackingView::~AreaTrackingView()
{
    //wai slam depends on _orbVocabulary and has to be uninitializd first
    _waiSlam.release();
    if (_voc)
        delete _voc;
    //_orbVocabulary.release();
}

bool AreaTrackingView::update()
{
    SENSFramePtr frame;
    if (_camera)
        frame = _camera->latestFrame();

    if (frame && _waiSlam)
    {
        //the intrinsics may change dynamically on focus changes (e.g. on iOS)
        if (!frame->intrinsics.empty())
        {
            auto    calib        = _camera->calibration();
            cv::Mat scaledCamMat = SENS::adaptCameraMat(_camera->calibration()->cameraMat(),
                                                        _camera->config().manipWidth,
                                                        _camera->config().targetWidth);
            _waiSlam->changeIntrinsic(scaledCamMat, calib->distortion());
            updateSceneCameraFov();
        }
        _waiSlam->update(frame->imgManip);

        if (_waiSlam->isTracking())
            _scene.updateCameraPose(_waiSlam->getPose());

        updateTrackingVisualization(_waiSlam->isTracking(), *frame.get());
    }

    return onPaint();
}

void AreaTrackingView::updateSceneCameraFov()
{
    //if the camera image height is smaller than the sceneview height,
    //we have to calculate the corresponding vertical field of view for the scene camera
    float imgWdivH = _camera->calibration()->imageAspectRatio();
    if (std::abs(this->scrWdivH() - imgWdivH) > 0.00001)
    {
        if (this->scrWdivH() > imgWdivH)
        {
            //bars left and right: directly use camera vertial field of view as scene vertical field of view
            _scene.updateCameraIntrinsics(_camera->calibration()->cameraFovVDeg());
        }
        else
        {
            //bars top and bottom: estimate vertical fov from cameras horizontal field of view and screen aspect ratio
            float fovV = SENS::calcFovDegFromOtherFovDeg(_camera->calibration()->cameraFovHDeg(), this->scrW(), this->scrH());
            _scene.updateCameraIntrinsics(fovV);
        }
    }
    else
    {
        //no bars because same aspect ration: directly use camera vertial field of view as scene vertical field of view
        _scene.updateCameraIntrinsics(_camera->calibration()->cameraFovVDeg());
    }
}

std::unique_ptr<WAIMap> AreaTrackingView::tryLoadMap(const std::string& slamMapFileName)
{
    std::unique_ptr<WAIMap> waiMap;
    if (!slamMapFileName.empty())
    {
        bool        mapFileExists = false;
        std::string mapFileName   = _erlebARDir + slamMapFileName;
        Utils::log("AreaTrackingView", "map file name for area is: %s", mapFileName.c_str());
        //hick hack for android: android extracts our .gz file and renames them without .gz
        if (Utils::fileExists(mapFileName))
        {
            mapFileExists = true;
        }
        else if (Utils::fileExists(mapFileName + ".gz"))
        {
            mapFileName += ".gz";
            mapFileExists = true;
        }
        else if (Utils::containsString(mapFileName, ".gz"))
        {
            std::string mapFileNameWOExt;

            size_t i = mapFileName.rfind('.gz');
            if (i != std::string::npos)
                mapFileNameWOExt = mapFileName.substr(0, i - 2);

            //try without .gz
            if (Utils::fileExists(mapFileNameWOExt))
            {
                mapFileName   = mapFileNameWOExt;
                mapFileExists = true;
            }
        }

        if (mapFileExists)
        {
            Utils::log("AreaTrackingView", "loading map file from: %s", mapFileName.c_str());
            WAIKeyFrameDB* keyframeDataBase = new WAIKeyFrameDB(_voc);
            waiMap                          = std::make_unique<WAIMap>(keyframeDataBase);
            bool mapLoadingSuccess          = WAIMapStorage::loadMap(waiMap.get(),
                                                            _scene.mapNode,
                                                            _voc,
                                                            mapFileName,
                                                            false,
                                                            true);
        }
    }

    return waiMap;
}

void AreaTrackingView::initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId)
{
    //todo:
    /*
    load correct map
    load correct model
    init correct extractors and number of levels (put it to ErlebAR::Area directly)
    move params to ErlebAR::Area
    start camera with size from area
    */
    ErlebAR::Location& location = _locations[locId];
    ErlebAR::Area&     area     = location.areas[areaId];
    _gui.initArea(area);

    //start camera
    if (!startCamera(area.cameraFrameTargetSize))
    {
        Utils::log("AreaTrackingView", "Could not start camera!");
        return;
    }

    //load model into scene graph
    _scene.rebuild(location.name, area.name);
    this->camera(_scene.cameraNode);
    updateSceneCameraFov();

    //initialize extractors
    _initializationExtractor = _featureExtractorFactory.make(area.initializationExtractorType, area.cameraFrameTargetSize, area.nExtractorLevels);
    _trackingExtractor       = _featureExtractorFactory.make(area.trackingExtractorType, area.cameraFrameTargetSize, area.nExtractorLevels);
    _relocalizationExtractor = _featureExtractorFactory.make(area.relocalizationExtractorType, area.cameraFrameTargetSize, area.nExtractorLevels);

    //load vocabulary
    std::string fileName = _vocabularyDir + _vocabularyFileName;
    if (Utils::fileExists(fileName))
    {
        Utils::log("AreaTrackingView", "loading voc file from: %s", fileName.c_str());
        _voc->loadFromFile(fileName);
    }

    //try to load map
    std::unique_ptr<WAIMap> waiMap = tryLoadMap(area.slamMapFileName);

    cv::Mat scaledCamMat = SENS::adaptCameraMat(_camera->calibration()->cameraMat(),
                                                _camera->config().manipWidth,
                                                _camera->config().targetWidth);

    WAISlam::Params params;
    params.cullRedundantPerc   = 0.95f;
    params.ensureKFIntegration = false;
    params.fixOldKfs           = true;
    params.onlyTracking        = false;
    params.retainImg           = false;
    params.serial              = false;
    params.trackOptFlow        = false;

    _waiSlam = std::make_unique<WAISlam>(
      scaledCamMat,
      _camera->calibration()->distortion(),
      _voc,
      _initializationExtractor.get(),
      _relocalizationExtractor.get(),
      _trackingExtractor.get(),
      std::move(waiMap),
      params);

    if (_trackingExtractor->doubleBufferedOutput())
        _imgBuffer.init(2, area.cameraFrameTargetSize);
    else
        _imgBuffer.init(1, area.cameraFrameTargetSize);
}

void AreaTrackingView::resume()
{
    startCamera(_cameraFrameResumeSize);
}

void AreaTrackingView::hold()
{
    _camera->stop();
}

bool AreaTrackingView::startCamera(const cv::Size& cameraFrameTargetSize)
{
    if (_camera)
    {
        if (_camera->started())
            _camera->stop();

        //we have to store this for a resume call..
        _cameraFrameResumeSize = cameraFrameTargetSize;

        int trackingImgW = 640;
        //float targetWdivH   = 4.f / 3.f;
        float targetWdivH   = (float)cameraFrameTargetSize.width / (float)cameraFrameTargetSize.height;
        int   aproxVisuImgW = 1000;
        int   aproxVisuImgH = (int)((float)aproxVisuImgW / targetWdivH);

        auto capProps   = _camera->captureProperties();
        auto bestConfig = capProps.findBestMatchingConfig(SENSCameraFacing::BACK, 65.f, aproxVisuImgW, aproxVisuImgH);

        if (bestConfig.first && bestConfig.second)
        {
            const SENSCameraDeviceProperties* const devProps     = bestConfig.first;
            const SENSCameraStreamConfig*           streamConfig = bestConfig.second;
            Utils::log("AreaTrackingView", "starting camera with stream config: w:%d h:%d", streamConfig->widthPix, streamConfig->heightPix);

            int cropW,
              cropH, w, h;
            SENS::calcCrop(cv::Size(streamConfig->widthPix, streamConfig->heightPix), targetWdivH, cropW, cropH, w, h);

            _camera->start(devProps->deviceId(),
                           *streamConfig,
                           cv::Size(w, h),
                           false,
                           false,
                           true,
                           trackingImgW,
                           true,
                           65.f);
        }
        else //try with unknown config (for desktop usage, there may be no high resolution available)
        {
            aproxVisuImgW    = 640;
            aproxVisuImgH    = (int)((float)aproxVisuImgW / targetWdivH);
            auto bestConfig2 = capProps.findBestMatchingConfig(SENSCameraFacing::UNKNOWN, 52.5f, aproxVisuImgW, aproxVisuImgH);
            if (bestConfig2.first && bestConfig2.second)
            {
                const SENSCameraDeviceProperties* const devProps     = bestConfig2.first;
                const SENSCameraStreamConfig*           streamConfig = bestConfig2.second;
                Utils::log("AreaTrackingView", "starting camera with stream config: w:%d h:%d", streamConfig->widthPix, streamConfig->heightPix);

                int cropW, cropH, w, h;
                SENS::calcCrop(cv::Size(streamConfig->widthPix, streamConfig->heightPix), targetWdivH, cropW, cropH, w, h);

                _camera->start(devProps->deviceId(),
                               *streamConfig,
                               cv::Size(w, h),
                               false,
                               false,
                               true,
                               trackingImgW,
                               true,
                               52.5f);
            }
        }

        return _camera->started();
    }
    else
    {
        return false;
    }
}

void AreaTrackingView::updateTrackingVisualization(const bool iKnowWhereIAm, SENSFrame& frame)
{
    //todo: add or remove crop in case of wide screens
    //undistort image and copy image to video texture
    _waiSlam->drawInfo(frame.imgRGB, frame.scaleToManip, true, _showKeyPoints, _showKeyPointsMatched);

    if (_camera->calibration()->state() == SENSCalibration::State::calibrated)
        _camera->calibration()->remap(frame.imgRGB, _imgBuffer.inputSlot());
    else
        _imgBuffer.inputSlot() = frame.imgRGB;

    //add bars to image instead of viewport adjustment (we update the mat in the buffer)
    //todo: the matrices in the buffer have different sizes.. problem? no! no!
    SENS::extendWithBars(_imgBuffer.outputSlot(), this->viewportWdivH());

    _scene.updateVideoImage(_imgBuffer.outputSlot());
    _imgBuffer.incrementSlot();

    //update map point visualization
    if (_showMapPC)
        _scene.renderMapPoints(_waiSlam->getMapPoints());
    else
        _scene.removeMapPoints();

    //update visualization of matched map points (when WAI pose is valid)
    if (_showMatchesPC && iKnowWhereIAm)
        _scene.renderMatchedMapPoints(_waiSlam->getMatchedMapPoints(_waiSlam->getLastFramePtr()));
    else
        _scene.removeMatchedMapPoints();
}
