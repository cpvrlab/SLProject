#include <views/AreaTrackingView.h>
#include <sens/SENSCamera.h>
#include <CVCalibration.h>
#include <WAIMapStorage.h>

AreaTrackingView::AreaTrackingView(sm::EventHandler&   eventHandler,
                                   SLInputManager&     inputManager,
                                   ErlebAR::Resources& resources,
                                   SENSCamera*         camera,
                                   int                 screenWidth,
                                   int                 screenHeight,
                                   int                 dotsPerInch,
                                   std::string         fontPath,
                                   std::string         imguiIniPath)
  : SLSceneView(nullptr, dotsPerInch, inputManager),
    _gui(eventHandler,
         resources,
         dotsPerInch,
         screenWidth,
         screenHeight,
         std::bind(&AppWAIScene::adjustAugmentationTransparency, &_scene, std::placeholders::_1),
         fontPath),
    _scene("AreaTrackingScene"),
    _camera(camera)
{
    scene(&_scene);
    init("AreaTrackingView", screenWidth, screenHeight, nullptr, nullptr, &_gui, imguiIniPath);
    //todo: ->moved to constructor of AreaTrackingScene: can this lead to any problems?
    //_scene.init();
    //_scene.build();
    onInitialize();
    //set scene camera into sceneview
    //(camera node not initialized yet)
    //this->camera(_scene.cameraNode);

    _locations = resources.locations();
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
    _gui.initArea(_locations[locId].areas[areaId]);
    //start camera
    startCamera();

    //load model into scene graph
    //todo: move standard nodes to a different function than model loading
    _scene.rebuild("", "");
    this->camera(_scene.cameraNode);

    //initialize extractors
    _initializationExtractor = _featureExtractorFactory.make(_initializationExtractorType, _cameraFrameTargetSize);
    _trackingExtractor       = _featureExtractorFactory.make(_trackingExtractorType, _cameraFrameTargetSize);
    if (_trackingExtractor->doubleBufferedOutput())
        _imgBuffer.init(2, _cameraFrameTargetSize);
    else
        _imgBuffer.init(1, _cameraFrameTargetSize);

    //load vocabulary
    _orbVocabulary = std::make_unique<ORB_SLAM2::ORBVocabulary>();
    if (Utils::fileExists(_vocabularyFileName))
        _orbVocabulary->loadFromBinaryFile(_vocabularyFileName);
    //load map
    //_keyframeDataBase = std::make_unique<WAIKeyFrameDB>(*_orbVocabulary.get());
    //_waiMap           = std::make_unique<WAIMap>(_keyframeDataBase.get());
    //if (!_mapFileName.empty())
    //{
    //    bool mapLoadingSuccess = WAIMapStorage::loadMap(_waiMap.get(),
    //                                                    _scene.mapNode,
    //                                                    _orbVocabulary.get(),
    //                                                    _mapFileName,
    //                                                    false,
    //                                                    true);
    //}
    //calibration
    _calibration = std::make_unique<CVCalibration>(_cameraFrameTargetSize, 65.f, false, false, CVCameraType::BACKFACING, Utils::ComputerInfos().get());

    //init wai slam
    _waiSlam = std::make_unique<WAISlam>(
      _calibration->cameraMat(),
      _calibration->distortion(),
      _orbVocabulary.get(),
      _initializationExtractor.get(),
      _trackingExtractor.get(),
      nullptr,
      false,
      false,
      false,
      0.95f);

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
    if (_camera && !_camera->started())
    {
        //start camera
        SENSCamera::Config config;
        config.targetWidth   = _cameraFrameTargetSize.width;
        config.targetHeight  = _cameraFrameTargetSize.height;
        config.convertToGray = true;

        _camera->init(SENSCamera::Facing::BACK);
        _camera->start(config);
    }
}

void AreaTrackingView::updateTrackingVisualization(const bool iKnowWhereIAm, cv::Mat& imgRGB)
{
    //undistort image and copy image to video texture
    _waiSlam->drawInfo(imgRGB, true, _showKeyPoints, _showKeyPointsMatched);

    if (_calibration->state() == CS_calibrated)
        _calibration->remap(imgRGB, _imgBuffer.inputSlot());
    else
        _imgBuffer.inputSlot() = imgRGB;

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
