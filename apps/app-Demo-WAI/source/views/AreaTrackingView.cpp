#include <views/AreaTrackingView.h>
#include <sens/SENSCamera.h>
#include <WAIMapStorage.h>
#include <sens/SENSUtils.h>

#define LOAD_ASYNC

AreaTrackingView::AreaTrackingView(sm::EventHandler&   eventHandler,
                                   SLInputManager&     inputManager,
                                   const ImGuiEngine&  imGuiEngine,
                                   ErlebAR::Resources& resources,
                                   SENSCamera*         camera,
                                   SENSGps*            gps,
                                   SENSOrientation*    orientation,
                                   const DeviceData&   deviceData)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    _gui(imGuiEngine,
         eventHandler,
         resources,
         deviceData.dpi(),
         deviceData.scrWidth(),
         deviceData.scrHeight(),
         std::bind(&AppWAIScene::adjustAugmentationTransparency, &_scene, std::placeholders::_1),
         deviceData.erlebARDir(),
         std::bind(&AreaTrackingView::getSimHelper, this)),
    _scene("AreaTrackingScene", deviceData.dataDir(), deviceData.erlebARDir()),
    _userGuidanceScene(deviceData.dataDir()),
    _gps(gps),
    _orientation(orientation),
    _resources(resources),
    _deviceData(deviceData),
    _userGuidance(&_userGuidanceScene, &_gui, _gps, _orientation, resources)
{
    scene(&_userGuidanceScene);
    init("AreaTrackingView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, &_gui, deviceData.writableDir());

    _camera = std::make_unique<SENSCvCamera>(camera);
    //todo: ->moved to constructor of AreaTrackingScene: can this lead to any problems?
    //_scene.init();
    //_scene.build();
    onInitialize();

    _locations = resources.locations();
}

AreaTrackingView::~AreaTrackingView()
{
    //wai slam depends on _orbVocabulary and has to be uninitializd first
    _waiSlam.reset();
    if (_voc)
        delete _voc;

    if (_asyncLoader)
        delete _asyncLoader;
}

SLbool AreaTrackingView::onMouseDown(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod)
{
    SLbool ret = SLSceneView::onMouseDown(button, scrX, scrY, mod);
    _gui.mouseDown(_gui.doNotDispatchMouse());
    return ret;
}

SLbool AreaTrackingView::onMouseMove(SLint x, SLint y)
{
    SLbool ret = SLSceneView::onMouseMove(x, y);
    _gui.mouseMove(_gui.doNotDispatchMouse());
    return ret;
}

void AreaTrackingView::onCameraParamsChanged()
{
    _waiSlam->changeIntrinsic(_camera->scaledCameraMat(), _camera->calibration()->distortion());
    updateSceneCameraFov();
}

bool AreaTrackingView::update()
{
    WAI::TrackingState slamState = WAI::TrackingState_None;
    try
    {
        SENSFramePtr frame;
        if (_camera)
            frame = _camera->latestFrame();

        bool isTracking = false;

        if (frame && _waiSlam)
        {
            //the intrinsics may change dynamically on focus changes (e.g. on iOS)
            if (!frame->intrinsics.empty())
            {
                _waiSlam->changeIntrinsic(_camera->scaledCameraMat(), _camera->calibration()->distortion());
                updateSceneCameraFov();
            }
            _waiSlam->update(frame->imgManip);
            slamState = _waiSlam->getTrackingState();

            isTracking = (_waiSlam->getTrackingState() == WAI::TrackingState_TrackingOK);
            if (isTracking)
                _scene.updateCameraPose(_waiSlam->getPose());
        }
        else if (_asyncLoader && _asyncLoader->isReady())
        {
            Utils::log("AreaTrackingView", "worker done");
            cv::Mat mapNodeOm = _asyncLoader->mapNodeOm();
            initSlam(mapNodeOm, _asyncLoader->moveWaiMap());

            delete _asyncLoader;
            _asyncLoader = nullptr;
            _userGuidance.dataIsLoading(false);
        }

        //switch between userguidance scene and tracking scene depending on tracking state
        VideoBackgroundCamera* currentCamera;
        if (isTracking)
        {
            this->scene(&_scene);
            this->camera(_scene.camera);
            currentCamera = _scene.camera;
        }
        else
        {
            this->scene(&_userGuidanceScene);
            this->camera(_userGuidanceScene.camera);
            currentCamera = _userGuidanceScene.camera;
        }

        //update visualization
        if (frame)
        {
            //decorate video image and update scene
            if (_waiSlam)
                updateTrackingVisualization(isTracking, *frame.get());

            //set video image camera background
            updateVideoImage(*frame.get(), currentCamera);
        }

        _userGuidance.updateTrackingState(isTracking);
        _userGuidance.updateSensorEstimations();
    }
    catch (std::exception& e)
    {
        _gui.showErrorMsg(e.what());
    }
    catch (...)
    {
        _gui.showErrorMsg("AreaTrackingView update: unknown exception catched!");
    }

    //render call
    return onPaint();
}

void AreaTrackingView::initSlam(const cv::Mat& mapNodeOm, std::unique_ptr<WAIMap> waiMap)
{
    if (!mapNodeOm.empty())
    {
        SLMat4f slOm = WAIMapStorage::convertToSLMat(mapNodeOm);
        //std::cout << "slOm: " << slOm.toString() << std::endl;
        _scene.mapNode->om(slOm);
    }

    WAISlam::Params params;
    params.cullRedundantPerc   = 0.95f;
    params.ensureKFIntegration = false;
    params.fixOldKfs           = true;
    params.onlyTracking        = false;
    params.retainImg           = false;
    params.serial              = false;
    params.trackOptFlow        = false;

    _waiSlam = std::make_unique<WAISlam>(
      _camera->scaledCameraMat(),
      _camera->calibration()->distortion(),
      _voc,
      _initializationExtractor.get(),
      _relocalizationExtractor.get(),
      _trackingExtractor.get(),
      std::move(waiMap),
      params);
}

std::unique_ptr<WAIMap> AreaTrackingView::tryLoadMap(const std::string& erlebARDir,
                                                     const std::string& slamMapFileName,
                                                     WAIOrbVocabulary*  voc,
                                                     cv::Mat&           mapNodeOm)
{
    std::unique_ptr<WAIMap> waiMap;
    if (!slamMapFileName.empty())
    {
        bool        mapFileExists = false;
        std::string mapFileName   = erlebARDir + slamMapFileName;
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

            size_t i = mapFileName.rfind(".gz");
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
            WAIKeyFrameDB* keyframeDataBase = new WAIKeyFrameDB(voc);
            waiMap                          = std::make_unique<WAIMap>(keyframeDataBase);

            bool mapLoadingSuccess = false;
            if (Utils::containsString(mapFileName, ".waimap"))
            {
                mapLoadingSuccess = WAIMapStorage::loadMapBinary(waiMap.get(),
                                                                 mapNodeOm,
                                                                 voc,
                                                                 mapFileName,
                                                                 false,
                                                                 true);
            }
            else
            {
                mapLoadingSuccess = WAIMapStorage::loadMap(waiMap.get(),
                                                           mapNodeOm,
                                                           voc,
                                                           mapFileName,
                                                           false,
                                                           true);
            }
        }
    }

    return waiMap;
}

void AreaTrackingView::initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId)
{
    _locId  = locId;
    _areaId = areaId;

    //stop possible wai slam instances
    if (_waiSlam)
        _waiSlam.reset();

    ErlebAR::Location& location = _locations[locId];
    ErlebAR::Area&     area     = location.areas[areaId];
    _gui.initArea(area);
    _userGuidance.areaSelected(area.id, area.llaPos, area.viewAngleDeg);

    //start camera
    if (!startCamera(area.cameraFrameTargetSize))
        Utils::log("AreaTrackingView", "Could not start camera!");

    //load model into scene graph
    _scene.rebuild(location.name, area.name);

    updateSceneCameraFov();

    //initialize extractors
    _initializationExtractor = _featureExtractorFactory.make(area.initializationExtractorType, area.cameraFrameTargetSize, area.nExtractorLevels);
    _trackingExtractor       = _featureExtractorFactory.make(area.trackingExtractorType, area.cameraFrameTargetSize, area.nExtractorLevels);
    _relocalizationExtractor = _featureExtractorFactory.make(area.relocalizationExtractorType, area.cameraFrameTargetSize, area.nExtractorLevels);

    if (_trackingExtractor->doubleBufferedOutput())
        _imgBuffer.init(2, area.cameraFrameTargetSize);
    else
        _imgBuffer.init(1, area.cameraFrameTargetSize);

    std::string vocFileName;
    if (area.vocFileName.empty())
        vocFileName = _deviceData.vocabularyDir() + _vocabularyFileName;
    else
        vocFileName = _deviceData.erlebARDir() + area.vocFileName;

#ifdef LOAD_ASYNC
    //delete managed object
    if (_asyncLoader)
        delete _asyncLoader;

    _asyncLoader = new MapLoader(_voc, area.vocLayer, vocFileName, _deviceData.erlebARDir(), area.slamMapFileName);
    _asyncLoader->start();
    _userGuidance.dataIsLoading(true);

#else
    if (Utils::fileExists(vocFileName))
    {
        Utils::log("AreaTrackingView", "loading voc file from: %s", vocFileName.c_str());
        _voc = new WAIOrbVocabulary();
        _voc->loadFromFile(vocFileName);
    }

    //try to load map
    cv::Mat                 mapNodeOm;
    std::unique_ptr<WAIMap> waiMap = tryLoadMap(_deviceData.erlebARDir(), area.slamMapFileName, _voc, mapNodeOm);
    //init slam
    initSlam(mapNodeOm, std::move(waiMap));
#endif
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
            _scene.camera->updateCameraIntrinsics(_camera->calibration()->cameraFovVDeg());
            _userGuidanceScene.camera->updateCameraIntrinsics(_camera->calibration()->cameraFovVDeg());
        }
        else
        {
            //bars top and bottom: estimate vertical fov from cameras horizontal field of view and screen aspect ratio
            float fovV = SENS::calcFovDegFromOtherFovDeg(_camera->calibration()->cameraFovHDeg(), this->scrW(), this->scrH());
            _scene.camera->updateCameraIntrinsics(fovV);
            _userGuidanceScene.camera->updateCameraIntrinsics(fovV);
        }
    }
    else
    {
        //no bars because same aspect ration: directly use camera vertial field of view as scene vertical field of view
        _scene.camera->updateCameraIntrinsics(_camera->calibration()->cameraFovVDeg());
        _userGuidanceScene.camera->updateCameraIntrinsics(_camera->calibration()->cameraFovVDeg());
    }
}

void AreaTrackingView::resume()
{
    if (_camera)
        _camera->start();
}

void AreaTrackingView::hold()
{
    _camera->stop();
}

bool AreaTrackingView::startCamera(const cv::Size& trackImgSize)
{
    if (_camera)
    {
        if (_camera->started())
            _camera->stop();

        if (_camera->supportsFacing(SENSCameraFacing::BACK)) //we are on android or ios. we can also expect high resolution support.
            _camera->configure(SENSCameraFacing::BACK, 1920, 1440, trackImgSize.width, trackImgSize.height, false, false, true);
        else
            _camera->configure(SENSCameraFacing::UNKNOWN, 640, 480, trackImgSize.width, trackImgSize.height, false, false, true);
        _camera->start();
        return _camera->started();
    }
    else
    {
        return false;
    }
}

void AreaTrackingView::updateVideoImage(SENSFrame& frame, VideoBackgroundCamera* videoBackground)
{
    if (_camera->calibration()->state() == SENSCalibration::State::calibrated)
        _camera->calibration()->remap(frame.imgBGR, _imgBuffer.inputSlot());
    else
        _imgBuffer.inputSlot() = frame.imgBGR;

    //add bars to image instead of viewport adjustment (we update the mat in the buffer)
    //todo: the matrices in the buffer have different sizes.. problem? no! no!
    SENS::extendWithBars(_imgBuffer.outputSlot(), this->viewportWdivH());

    //update the current scene
    videoBackground->updateVideoImage(_imgBuffer.outputSlot());

    _imgBuffer.incrementSlot();
}

void AreaTrackingView::updateTrackingVisualization(const bool iKnowWhereIAm, SENSFrame& frame)
{
    //todo: add or remove crop in case of wide screens
    //undistort image and copy image to video texture
    if (_resources.developerMode)
        _waiSlam->drawInfo(frame.imgBGR, frame.scaleToManip, true, false, true);

    //update map point visualization
    if (_resources.developerMode)
        _scene.renderMapPoints(_waiSlam->getMapPoints());

    //update visualization of matched map points (when WAI pose is valid)
    if (iKnowWhereIAm && (_gui.opacity() > 0.0001f))
    {
        auto lastFrame = _waiSlam->getLastFrame();
        _scene.renderMatchedMapPoints(_waiSlam->getMatchedMapPoints(&lastFrame), _gui.opacity());
    }
    else
        _scene.removeMatchedMapPoints();
}
