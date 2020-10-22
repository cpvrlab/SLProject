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
         std::bind(&AppWAIScene::adjustAugmentationTransparency, &_waiScene, std::placeholders::_1),
         deviceData.erlebARDir(),
         std::bind(&AreaTrackingView::getSimHelper, this)),
    _waiScene("AreaTrackingScene", deviceData.dataDir(), deviceData.erlebARDir()),
    _userGuidanceScene(deviceData.dataDir()),
    _gps(gps),
    _orientation(orientation),
    _resources(resources),
    _deviceData(deviceData),
    _userGuidance(&_userGuidanceScene, &_gui, _gps, _orientation, resources),
    _locations(resources.locations())
{
    scene(&_userGuidanceScene);
    init("AreaTrackingView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, &_gui, deviceData.writableDir());
    onInitialize();

    //init video camera
    _camera = std::make_unique<SENSCvCamera>(camera);
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

void AreaTrackingView::initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId)
{
    try
    {
        _noInitException = false;
        _locId                      = locId;
        _areaId                     = areaId;
        ErlebAR::Location& location = _locations[locId];
        ErlebAR::Area&     area     = location.areas[areaId];

        //stop and reset possible wai slam instances
        if (_waiSlam)
            _waiSlam.reset();

        _gui.initArea(area);
        if (_resources.enableUserGuidance)
            _userGuidance.areaSelected(area.id, area.llaPos, area.viewAngleDeg);

        //start video camera
        startCamera(area.cameraFrameTargetSize);

        //init 3d visualization
        this->unInit();
        _waiScene.initScene(locId, areaId);
        updateSceneCameraFov();
        this->onInitialize(); //init scene view

        initDeviceLocation(location, area);
        initSlam(area);
        
        _noInitException = true;
    }
    catch (std::exception& e)
    {
        _gui.showErrorMsg(e.what());
    }
    catch (...)
    {
        _gui.showErrorMsg("AreaTrackingView init: unknown exception catched!");
    }
}

bool AreaTrackingView::update()
{
    WAI::TrackingState slamState = WAI::TrackingState_None;
    try
    {
        if (_noInitException)
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
                    _waiScene.updateCameraPose(_waiSlam->getPose());
                else if (_orientation)
                {
                    SLMat4f camPose = calcCameraPoseGpsOrientationBased();
                    _waiScene.camera->om(camPose);
                    //give waiSlam a guess of the current position in the ENU frame
                    //todo ..
                }
            }
            else if (_asyncLoader && _asyncLoader->isReady())
            {
                Utils::log("AreaTrackingView", "worker done");
                cv::Mat mapNodeOm = _asyncLoader->mapNodeOm();
                initWaiSlam(mapNodeOm, _asyncLoader->moveWaiMap());

                delete _asyncLoader;
                _asyncLoader = nullptr;
                if (_resources.enableUserGuidance)
                    _userGuidance.dataIsLoading(false);
            }

            //switch between userguidance scene and tracking scene depending on tracking state
            VideoBackgroundCamera* currentCamera;
            if (isTracking || !_resources.enableUserGuidance)
            {
                this->scene(&_waiScene);
                this->camera(_waiScene.camera);
                currentCamera = _waiScene.camera;
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

            if (_resources.enableUserGuidance)
            {
                _userGuidance.updateTrackingState(isTracking);
                _userGuidance.updateSensorEstimations();
            }
        }
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

//call when view becomes visible
void AreaTrackingView::onShow()
{
    _gui.onShow();
    if (_gps)
        _gps->start();
    if (_orientation)
        _orientation->start();

    if (_resources.developerMode && _resources.simulatorMode)
    {
        if (_simHelper)
            _simHelper.reset();
        _simHelper = std::make_unique<SENSSimHelper>(_gps,
                                                     _orientation,
                                                     _camera->cameraRef(),
                                                     _deviceData.writableDir() + "SENSSimData",
                                                     std::bind(&AreaTrackingView::onCameraParamsChanged, this));
    }
}

void AreaTrackingView::onHide()
{
    //reset user guidance and run it once
    _userGuidance.reset();

    if (_gps)
        _gps->stop();
    if (_orientation)
        _orientation->stop();
    if (_camera)
        _camera->stop();

    if (_simHelper)
        _simHelper.reset();
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

SLMat4f AreaTrackingView::calcCameraPoseGpsOrientationBased()
{
    //use gps and orientation sensor for camera position and orientation
    //(even if there is no gps, devLoc gives us a guess of the current home position)
    if (_gps)
    {
        //update with current gps sensor position
        auto loc = _gps->getLocation();
        _devLoc.onLocationLatLonAlt(loc.latitudeDEG, loc.longitudeDEG, loc.altitudeM, loc.altitudeM);
    }

    auto     sensQuat = _orientation->getOrientation();
    SLQuat4f slQuat(sensQuat.quatX, sensQuat.quatY, sensQuat.quatZ, sensQuat.quatW);
    SLMat3f  rotMat = slQuat.toMat3();

    SLMat4f camPose;
    {
        SLMat3f sRc;
        sRc.rotation(-90, 0, 0, 1);

        //sensor rotation w.r.t. east-north-down
        SLMat3f enuRs;
        enuRs.setMatrix(rotMat);

        //world-yaw rotation w.r.t. world
        SLMat3f wRwy;
        wRwy.rotation(-90, 1, 0, 0);

        //combiniation of partial rotations to orientation of camera w.r.t world
        SLMat3f wRc = wRwy * enuRs * sRc;
        camPose.setRotation(wRc);
    }

    //The scene must have a global reference position
    if (_devLoc.hasOrigin())
    {
        // Direction vector from camera to world origin
        SLVec3d wtc = _devLoc.locENU() - _devLoc.originENU();

        // Reset to default if device is too far away
        if (wtc.length() > _devLoc.locMaxDistanceM())
            wtc = _devLoc.defaultENU() - _devLoc.originENU();

        // Set the camera position
        SLVec3f wtc_f((SLfloat)wtc.x, (SLfloat)wtc.y, (SLfloat)wtc.z);
        camPose.setTranslation(wtc_f);
    }

    return camPose;
}

void AreaTrackingView::initSlam(const ErlebAR::Area& area)
{
    //initialize extractors
    _initializationExtractor = _featureExtractorFactory.make(area.initializationExtractorType, area.cameraFrameTargetSize, area.nExtractorLevels);
    _trackingExtractor       = _featureExtractorFactory.make(area.trackingExtractorType, area.cameraFrameTargetSize, area.nExtractorLevels);
    _relocalizationExtractor = _featureExtractorFactory.make(area.relocalizationExtractorType, area.cameraFrameTargetSize, area.nExtractorLevels);

    if (_trackingExtractor->doubleBufferedOutput())
        _imgBuffer.init(2, area.cameraFrameTargetSize);
    else
        _imgBuffer.init(1, area.cameraFrameTargetSize);

    std::string vocFileName = _deviceData.dataDir() + area.vocFileName;

#ifdef LOAD_ASYNC
    //delete managed object
    if (_asyncLoader)
        delete _asyncLoader;

    _asyncLoader = new MapLoader(_voc, area.vocLayer, vocFileName, _deviceData.erlebARDir(), area.slamMapFileName);
    _asyncLoader->start();
    if (_resources.enableUserGuidance)
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
    initWaiSlam(mapNodeOm, std::move(waiMap));
#endif
}

void AreaTrackingView::initWaiSlam(const cv::Mat& mapNodeOm, std::unique_ptr<WAIMap> waiMap)
{
    if (!mapNodeOm.empty())
    {
        SLMat4f slOm = WAIMapStorage::convertToSLMat(mapNodeOm);
        //std::cout << "slOm: " << slOm.toString() << std::endl;
        _waiScene.mapNode->om(slOm);
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

void AreaTrackingView::initDeviceLocation(const ErlebAR::Location& location, const ErlebAR::Area& area)
{
    //reset everything to default
    _devLoc.init();

    _devLoc.locMaxDistanceM(1000.0f); // Max. Distanz. to Area home
    _devLoc.improveOrigin(false);     // Keine autom. Verbesserung vom Origin
    _devLoc.useOriginAltitude(true);
    _devLoc.hasOrigin(true);
    // Let the sun be rotated by time and location
    if (_waiScene.sunLight)
        _devLoc.sunLightNode(_waiScene.sunLight);

    _devLoc.originLatLonAlt(area.modelOrigin.x, area.modelOrigin.y, area.modelOrigin.z); // Model origin
    _devLoc.defaultLatLonAlt(area.llaPos.x, area.llaPos.y, area.llaPos.z + 1.7);
    //ATTENTION: call this after originLatLonAlt and defaultLatLonAlt setters. Otherwise alititude will be overwritten!!
    if (!location.geoTiffFileName.empty())
    {
        std::string geoTiffFileName = _deviceData.erlebARDir() + location.geoTiffFileName;
        if (Utils::fileExists(geoTiffFileName))
            _devLoc.loadGeoTiff(geoTiffFileName, "ErlebAR");
        else
        {
            std::stringstream ss;
            ss << "AreaTrackingView::initDeviceLocation: geo tiff file does not exist: " << geoTiffFileName;
            throw std::runtime_error(ss.str());
        }
    }
    else
        Utils::log("AreaTrackingView", "WARNING: no geo tiff available");
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
            _waiScene.camera->updateCameraIntrinsics(_camera->calibration()->cameraFovVDeg());
            _userGuidanceScene.camera->updateCameraIntrinsics(_camera->calibration()->cameraFovVDeg());
        }
        else
        {
            //bars top and bottom: estimate vertical fov from cameras horizontal field of view and screen aspect ratio
            float fovV = SENS::calcFovDegFromOtherFovDeg(_camera->calibration()->cameraFovHDeg(), this->scrW(), this->scrH());
            _waiScene.camera->updateCameraIntrinsics(fovV);
            _userGuidanceScene.camera->updateCameraIntrinsics(fovV);
        }
    }
    else
    {
        //no bars because same aspect ration: directly use camera vertial field of view as scene vertical field of view
        _waiScene.camera->updateCameraIntrinsics(_camera->calibration()->cameraFovVDeg());
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
        Utils::log("AreaTrackingView", "Could not start camera!");
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
        _waiScene.renderMapPoints(_waiSlam->getMapPoints());

    //update visualization of matched map points (when WAI pose is valid)
    if (iKnowWhereIAm && (_gui.opacity() > 0.0001f))
    {
        auto lastFrame = _waiSlam->getLastFrame();
        _waiScene.renderMatchedMapPoints(_waiSlam->getMatchedMapPoints(&lastFrame), _gui.opacity());
    }
    else
        _waiScene.removeMatchedMapPoints();
}
