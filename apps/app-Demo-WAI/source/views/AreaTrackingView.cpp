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
    //_orbVocabulary.release();
}

bool AreaTrackingView::update()
{
    SENSFramePtr frame;
    if (_camera)
        frame = _camera->getLatestFrame();

    if (frame && _waiSlam)
    {
        _waiSlam->update(frame->imgGray);

        if (_waiSlam->isTracking())
            _scene.updateCameraPose(_waiSlam->getPose());

        updateTrackingVisualization(_waiSlam->isTracking(), frame->imgRGB);
    }

    return onPaint();
}

void AreaTrackingView::initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId)
{
    ErlebAR::Location& location = _locations[locId];
    ErlebAR::Area&     area     = location.areas[areaId];
    _gui.initArea(area);
    //start camera
    startCamera();

    //load model into scene graph
    //todo: move standard nodes to a different function than model loading
    //doWaitOnIdle(false);
    _scene.rebuild(location.name, area.name);
    this->camera(_scene.cameraNode);
    //onInitialize();
    //setViewportFromRatio(SLVec2i(_camera->getFrameSize().width, _camera->getFrameSize().height), SLViewportAlign::VA_center, true);

    //calibration
    const SENSCameraCharacteristics& chars = _camera->characteristics();
    if (chars.provided)
    {
        _calibration = std::make_unique<SENSCalibration>(chars.physicalSensorSizeMM.width,
                                                         chars.physicalSensorSizeMM.height,
                                                         chars.focalLenghtsMM.front(),
                                                         _cameraFrameTargetSize,
                                                         false,
                                                         false,
                                                         SENSCameraType::BACKFACING,
                                                         Utils::ComputerInfos().get());
    }
    else
    {
        //params from webcam calibration file
        cv::Size calibImgSize(1600, 896);
        _calibration = std::make_unique<SENSCalibration>(_cameraFrameTargetSize, 69.59405517578125f, false, false, SENSCameraType::BACKFACING, Utils::ComputerInfos().get());
        //todo:
        //the calculated fov vertical does not fit to the one of the calibration file->normal ?
    }

    //cv::Size calibImgSize(3968, 2976);
    //_calibration = std::make_unique<SENSCalibration>(calibImgSize, 60.42 /*63.144f*/, false, false, SENSCameraType::BACKFACING, Utils::ComputerInfos().get());

    //std::string calibDir      = "C:/Users/ghm1/AppData/Roaming/SLProject/calibrations/";
    //std::string calibFileName = "camCalib_ghm1-DESKTOP-V8HAA50-MODEL-_main.xml";
    //if (!_calibration->load(calibDir, Utils::getFileName(calibFileName), false))
    //{
    //    //_gui.showErrorMsg("Error when loading calibration from file: " +
    //    //                  slamParams.calibrationFile);
    //    return;
    //}

    if (_calibration->imageSize() != _cameraFrameTargetSize)
    {
        _calibration->adaptForNewResolution(_cameraFrameTargetSize, true);
    }

    //todo:
    ////parameterize scene camera from calibration
    //cv::Mat m  = _calibration->cameraMatUndistorted();
    //double  fx = (float)m.at<double>(0, 0);
    //double  fy = (float)m.at<double>(1, 1);
    //double  cx = (float)m.at<double>(0, 2);
    //double  cy = (float)m.at<double>(1, 2);
    //m          = (cv::Mat_<double>(3, 3) << fx, 0, cx + 106.f, 0, fy, cy, 0, 0, 1);
    _scene.updateCameraIntrinsics(_calibration->cameraFovVDeg(), _calibration->cameraMatUndistorted());

    //initialize extractors
    _initializationExtractor = _featureExtractorFactory.make(_initializationExtractorType, _cameraFrameTargetSize);
    _trackingExtractor       = _featureExtractorFactory.make(_trackingExtractorType, _cameraFrameTargetSize);

    //load vocabulary
    std::string fileName = _vocabularyDir + _vocabularyFileName;
    if (Utils::fileExists(fileName))
    {
        Utils::log("AreaTrackingView", "loading voc file from: %s", fileName.c_str());
        _voc->loadFromFile(fileName);
    }

    //load map
    std::unique_ptr<WAIMap> waiMap;
    if (!area.slamMapFileName.empty())
    {
        Utils::log("AreaTrackingView", "map for area is: %s", area.slamMapFileName.c_str());
        std::string mapFileName = _erlebARDir + area.slamMapFileName;
        if (Utils::fileExists(mapFileName))
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

    //init wai slam
    _waiSlam = std::make_unique<WAISlam>(
      _calibration->cameraMat(),
      _calibration->distortion(),
      _voc,
      _initializationExtractor.get(),
      _trackingExtractor.get(),
      std::move(waiMap),
      false,
      false,
      false,
      0.95f);

    //setViewportFromRatio(SLVec2i(_cameraFrameTargetSize.width, _cameraFrameTargetSize.height), SLViewportAlign::VA_center, true);
    //todo: adjust field of view (or intrinsics) so it fits the bars

    if (_trackingExtractor->doubleBufferedOutput())
        _imgBuffer.init(2, _cameraFrameTargetSize);
    else
        _imgBuffer.init(1, _cameraFrameTargetSize);

    //start wai with map for this area (as non-blocking as possible)
    //todo: separate loading from opengl calls (task in projectplan)
}

void AreaTrackingView::resume()
{
    startCamera();
}

void AreaTrackingView::hold()
{
    _camera->stop();
}

void AreaTrackingView::startCamera()
{
    if (_camera)
    {
        if (_camera->started())
            _camera->stop();

        //start camera
        SENSCameraConfig config;
        config.targetWidth   = _cameraFrameTargetSize.width;
        config.targetHeight  = _cameraFrameTargetSize.height;
        config.convertToGray = true;

        _camera->start(config);
    }
}

void AreaTrackingView::updateTrackingVisualization(const bool iKnowWhereIAm, cv::Mat& imgRGB)
{
    //undistort image and copy image to video texture
    _waiSlam->drawInfo(imgRGB, true, _showKeyPoints, _showKeyPointsMatched);

    if (_calibration->state() == SENSCalibration::State::calibrated)
        _calibration->remap(imgRGB, _imgBuffer.inputSlot());
    else
        _imgBuffer.inputSlot() = imgRGB;

    //add bars to image instead of viewport adjustment (we update the mat in the buffer)
    //todo: the matrices in the buffer have different sizes.. problem? no! no!
    int addW, addH;
    SENS::extendWithBars(_imgBuffer.outputSlot(), this->viewportWdivH(), cv::BORDER_REPLICATE, addW, addH);

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
