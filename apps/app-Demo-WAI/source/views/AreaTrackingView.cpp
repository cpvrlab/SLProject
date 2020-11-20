#include <views/AreaTrackingView.h>
#include <sens/SENSCamera.h>
#include <WAIMapStorage.h>
#include <sens/SENSUtils.h>
#include <Utils.h>
#include <GlobalTimer.h>

#define LOAD_ASYNC
//#define TARGET_WIDTH 1920
//#define TARGET_HEIGHT 1440
#define TARGET_WIDTH 640
#define TARGET_HEIGHT 360

AreaTrackingView::AreaTrackingView(sm::EventHandler&  eventHandler,
                                   SLInputManager&    inputManager,
                                   const ImGuiEngine& imGuiEngine,
                                   ErlebAR::Config&   config,
                                   SENSCamera*        camera,
                                   SENSGps*           gps,
                                   SENSOrientation*   orientation,
                                   SENSARCore*        arcore,
                                   const DeviceData&  deviceData)
  : SLSceneView(nullptr, deviceData.dpi(), inputManager),
    _gui(imGuiEngine,
         eventHandler,
         config,
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
    _arcore(arcore),
    _config(config),
    _resources(config.resources()),
    _deviceData(deviceData),
    _userGuidance(&_userGuidanceScene, &_gui, _gps, _orientation, config.resources()),
    _locations(config.locations())
{
    scene(&_userGuidanceScene);
    this->camera(_userGuidanceScene.camera);

    init("AreaTrackingView", deviceData.scrWidth(), deviceData.scrHeight(), nullptr, nullptr, &_gui, deviceData.writableDir());
    onInitialize();

    //init video camera
    _camera = std::make_unique<SENSCvCamera>(camera);
}

AreaTrackingView::~AreaTrackingView()
{
}

void AreaTrackingView::initArea(ErlebAR::LocationId locId, ErlebAR::AreaId areaId)
{
    try
    {
        _noInitException = false;
        _gui.showErrorMsg("");

        _locId                      = locId;
        _areaId                     = areaId;
        ErlebAR::Location& location = _locations[locId];
        ErlebAR::Area&     area     = location.areas[areaId];

        //stop and reset possible wai slam instances
        if (_waiSlam)
            _waiSlam.reset();

        if (_arcore && _config.useARCore)
            _arcore->reset();

        _gui.initArea(area);
        if (_config.enableUserGuidance)
            _userGuidance.areaSelected(area.id, area.llaPos, area.viewAngleDeg);

        //start video camera
        startCamera(area.cameraFrameTargetSize);

        //init 3d visualization
        this->unInit();
        _waiScene.initScene(locId, areaId);
        updateSceneCameraFov();
        this->scene(&_waiScene);
        this->camera(_waiScene.camera);
        this->onInitialize(); //init scene view

        //init arcore
        if (_arcore && _config.useARCore && _arcore->isAvailable())
        {
            _arcore->init(TARGET_WIDTH, TARGET_HEIGHT, area.cameraFrameTargetSize.width, area.cameraFrameTargetSize.height, true);
            _arcore->resume();
        }

        initDeviceLocation(location, area);
        initSlam(area);

        _initTime = GlobalTimer::timeS();
        _frameCounter = 0;
        _hasTransitionMatrix = false;
        _transitionMatrix.identity();
        //_avgPose.init(20, SLMat4f());

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

cv::Mat AreaTrackingView::convertCameraPoseToWaiCamExtrinisc(SLMat4f& wTc)
{
    // update camera node position
    cv::Mat wRc(3, 3, CV_32F);
    cv::Mat wtc(3, 1, CV_32F);
    //wTc.print("wtc");

    //copy from SLMat4 to cv::Mat rotation and translation and invert sign of y- and z-axis
    // clang-format off
    wRc.at<float>(0,0) = wTc.m(0); wRc.at<float>(0,1) = -wTc.m(4); wRc.at<float>(0,2) = -wTc.m(8); wtc.at<float>(0) = wTc.m(12);
    wRc.at<float>(1,0) = wTc.m(1); wRc.at<float>(1,1) = -wTc.m(5); wRc.at<float>(1,2) = -wTc.m(9); wtc.at<float>(1) = wTc.m(13);
    wRc.at<float>(2,0) = wTc.m(2); wRc.at<float>(2,1) = -wTc.m(6); wRc.at<float>(2,2) = -wTc.m(10); wtc.at<float>(2) = wTc.m(14);
    // clang-format on
    //std::cout << "wRc: " << wRc << std::endl;
    //std::cout << "wtc: " << wtc << std::endl;

    //inversion of orthogonal rotation matrix
    cv::Mat cRw = wRc.t();
    //inversion of vector
    cv::Mat ctw = -cRw * wtc;
    //std::cout << "cRw: " << cRw << std::endl;
    //std::cout << "ctw: " << ctw << std::endl ;

    //copy to 4x4 matrix
    cv::Mat cTw = cv::Mat::eye(4, 4, CV_32F);
    cRw.copyTo(cTw.colRange(0, 3).rowRange(0, 3));
    ctw.copyTo(cTw.rowRange(0, 3).col(3));
    //std::cout << "cTw: " << cTw << std::endl;
    return cTw;
}

/*
 cTw is the camera extrinsic: the world w.r.t. the camera coordinate system.
 We invert to get the camera pose (camera w.r.t the world coordinate system).
 */
SLMat4f convertARCoreToSLMat(const cv::Mat& cTw)
{
    SLMat4f m;

    // update camera node position
    cv::Mat wRc(3, 3, CV_32F);
    cv::Mat wtc(3, 1, CV_32F);

    //inversion of orthogonal rotation matrix
    wRc = (cTw.rowRange(0, 3).colRange(0, 3)).t();
    //inversion of vector
    wtc = -wRc * cTw.rowRange(0, 3).col(3);

    cv::Mat wTc = cv::Mat::eye(4, 4, CV_32F);
    wRc.copyTo(wTc.colRange(0, 3).rowRange(0, 3));
    wtc.copyTo(wTc.rowRange(0, 3).col(3));

    // clang-format off
    m.setMatrix(-wTc.at<float>(0, 1), wTc.at<float>(0, 0), wTc.at<float>(0, 2), wTc.at<float>(0, 3),
                -wTc.at<float>(1, 1), wTc.at<float>(1, 0), wTc.at<float>(1, 2), wTc.at<float>(1, 3),
                -wTc.at<float>(2, 1), wTc.at<float>(2, 0), wTc.at<float>(2, 2), wTc.at<float>(2, 3),
                -wTc.at<float>(3, 1), wTc.at<float>(3, 0), wTc.at<float>(3, 2), wTc.at<float>(3, 3));
    // clang-format on
    return m;
}

SLMat4f convertWAISlamToSLMat(const cv::Mat& cTw)
{
    SLMat4f m;

    // update camera node position
    cv::Mat wRc(3, 3, CV_32F);
    cv::Mat wtc(3, 1, CV_32F);

    //inversion of orthogonal rotation matrix
    wRc = (cTw.rowRange(0, 3).colRange(0, 3)).t();
    //inversion of vector
    wtc = -wRc * cTw.rowRange(0, 3).col(3);

    cv::Mat wTc = cv::Mat::eye(4, 4, CV_32F);
    wRc.copyTo(wTc.colRange(0, 3).rowRange(0, 3));
    wtc.copyTo(wTc.rowRange(0, 3).col(3));

    // clang-format off
    //set and invert y and z axes
    m.setMatrix(wTc.at<float>(0, 0), -wTc.at<float>(0, 1), -wTc.at<float>(0, 2), wTc.at<float>(0, 3),
                wTc.at<float>(1, 0), -wTc.at<float>(1, 1), -wTc.at<float>(1, 2), wTc.at<float>(1, 3),
                wTc.at<float>(2, 0), -wTc.at<float>(2, 1), -wTc.at<float>(2, 2), wTc.at<float>(2, 3),
                wTc.at<float>(3, 0), -wTc.at<float>(3, 1), -wTc.at<float>(3, 2), wTc.at<float>(3, 3));
    // clang-format on
    return m;
}

bool AreaTrackingView::updateGPSARCore(SENSFramePtr& frame)
{
    cv::Mat view;
    bool    isTracking = false;

    SLMat4f gpsPose = calcCameraPoseGpsOrientationBased();

    //Get frame from ArCore or camera depending if we have them.
    if (_arcore->isRunning())
    {
        cv::Mat proj;
        isTracking = _arcore->update(proj, view);
        frame      = _arcore->latestFrame();
    }
    else if (_camera)
        frame = _camera->latestFrame();
    else
        return false;

    _waiScene.camera->om(gpsPose);

    if (!_hasTransitionMatrix)
    {
        _gui.showInfoText("GPS + Sensors positioning");
        _waiScene.camera->om(gpsPose);
        //delay arcore transition unil 10s and that arcore is tracking at this state

        /*
        if (GlobalTimer::timeS() - _initTime > 6.0)
        {
            _avgPose.set(gpsPose);
            //_avgPose += gpsPose;
        }
        */
//        else
//            _avgPose.init(20, SLMat4f());

        if (GlobalTimer::timeS() - _initTime > 10.0 && isTracking)
        {
            _transitionMatrix = convertARCoreToSLMat(view);
            _transitionMatrix.invert();
            _transitionMatrix    = gpsPose * _transitionMatrix;
            //_transitionMatrix    = _avgPose.average() * _transitionMatrix;
            _hasTransitionMatrix = true;
        }
    }
    else if (_arcore->isRunning() && isTracking)
    {
        _gui.showInfoText("ARCore tracking");
        SLMat4f arPose  = convertARCoreToSLMat(view);
        SLMat4f camPose = _transitionMatrix * arPose;
        applyFingerCorrection(camPose);
        _waiScene.camera->om(camPose);
    }

    return isTracking;
}

bool AreaTrackingView::updateGPSWAISlamARCore(SENSFramePtr& frame)
{
    cv::Mat view;
    bool    isTracking = false;

    SLMat4f gpsPose = calcCameraPoseGpsOrientationBased();

    if (_arcore->isRunning())
    {
        cv::Mat proj;
        isTracking = _arcore->update(proj, view);
        frame      = _arcore->latestFrame();
    }
    else if (_camera)
    {
        _gui.showInfoText("arcore not running");
        frame = _camera->latestFrame();
    }
    else
    {
        _gui.showInfoText("arcore not running");
        return false;
    }

    _waiScene.camera->om(gpsPose);

    // Try to relocalize with WAISlam if there is a map for the area, otherwise fallback to GPS
    if (!_hasTransitionMatrix)
    {
        //_gui.showInfoText("GPS + Sensors positioning");
        if (frame && _waiSlam && _waiSlam->isInitialized() && isTracking) // TODO: Add timing condition to fallback to GPS if WAISlam too slow
        {
            //the intrinsics may change dynamically on focus changes (e.g. on iOS)
            //if (!frame->intrinsics.empty())
            //{
            //_waiSlam->changeIntrinsic(_camera->scaledCameraMat(), _camera->calibration()->distortion());
            //updateSceneCameraFov();
            //}

            //(not used yet, so I commented it for now)
            //cv::Mat camExtrinsic = convertCameraPoseToWaiCamExtrinisc(gpsPose);
            //_waiSlam->setCamExrinsicGuess(camExtrinsic);

            _waiSlam->update(frame->imgManip);

            if (WAI::TrackingState_TrackingOK == _waiSlam->getTrackingState())
            {
                _frameCounter++;
                if (_frameCounter > 20)
                {
                    _transitionMatrix = convertARCoreToSLMat(view);
                    _transitionMatrix.invert();
                    _transitionMatrix    = convertWAISlamToSLMat(_waiSlam->getPose()) * _transitionMatrix;
                    _hasTransitionMatrix = true;
                    _gui.showInfoText("WAISlam -> ARCore");
                }
            }
            else
            {
                _frameCounter = 0;
                _gui.showInfoText("WAISlam is not tracking");
            }
        }
        else if (GlobalTimer::timeS() - _initTime > 10.0 && isTracking)
        {
            _transitionMatrix = convertARCoreToSLMat(view);
            _transitionMatrix.invert();
            _transitionMatrix    = gpsPose * _transitionMatrix;
            _hasTransitionMatrix = true;
        }
        else
        {
            if (isTracking)
                _gui.showInfoText("no transition matrix yet, arcore is tracking");
            else
                _gui.showInfoText("no transition matrix yet, arcore is not tracking");
        }
    }
    else if (_arcore->isRunning() && isTracking)
    {
        _gui.showInfoText("ARCore tracking");
        SLMat4f arPose  = convertARCoreToSLMat(view);
        SLMat4f camPose = _transitionMatrix * arPose;
        applyFingerCorrection(camPose);
        _waiScene.camera->om(camPose);
    }
    else
    {
        _gui.showInfoText("GPS + sensors tracking");
    }
    //else {} TODO: if waiSlam was working, try to reloc with waislam after some time
    return isTracking;
}

bool AreaTrackingView::updateGPSWAISlam(SENSFramePtr& frame)
{
    bool isTracking = false;

    cv::Mat view;
    if (_arcore->isRunning())
    {
        cv::Mat proj;
        isTracking = _arcore->update(proj, view);
        frame      = _arcore->latestFrame();
    }
    else if (_camera)
        frame = _camera->latestFrame();
    else
        return false;

    if (frame && _waiSlam)
    {
        if (_waiSlam->isInitialized())
        {
            //the intrinsics may change dynamically on focus changes (e.g. on iOS)
            if (!frame->intrinsics.empty())
            {
                _waiSlam->changeIntrinsic(_camera->scaledCameraMat(), _camera->calibration()->distortion());
                updateSceneCameraFov();
            }
            _waiSlam->update(frame->imgManip);
            isTracking = (_waiSlam->getTrackingState() == WAI::TrackingState_TrackingOK);
            updateTrackingVisualization(isTracking, *frame.get());
        }
        else
        {
            _gui.showInfoText("waislam not initialized");
        }

        if (isTracking)
        {
            _gui.showInfoText("waislam");
            _waiScene.updateCameraPose(_waiSlam->getPose());
        }
    }

    //fallback to gps and orientation sensor
    if (!isTracking && _orientation)
    {
        _gui.showInfoText("no tracking -> gps");
        //use gps and orientation sensor for camera position and orientation
        //(even if there is no gps, devLoc gives us a guess of the current home position)
        SLMat4f gpsPose = calcCameraPoseGpsOrientationBased();
        applyFingerCorrection(gpsPose);
        _waiScene.camera->om(gpsPose);

        //give waiSlam a guess of the current position in the ENU frame
        //cv::Mat camExtrinsic = convertCameraPoseToWaiCamExtrinisc(gpsPose);
        //_waiSlam->setCamExrinsicGuess(camExtrinsic);
    }

    return isTracking;
}

// For marker initialization without arcore
bool AreaTrackingView::updateWAISlamGPS(SENSFramePtr& frame)
{
    cv::Mat view;
    bool    isTracking = false;

    SLMat4f gpsPose = calcCameraPoseGpsOrientationBased();

    if (_arcore->isRunning())
    {
        cv::Mat proj;
        isTracking = _arcore->update(proj, view);
        frame      = _arcore->latestFrame();
    }
    else if (_camera)
        frame = _camera->latestFrame();
    else
        return false;

    _waiScene.camera->om(gpsPose);

    if (frame && _waiSlam && _waiSlam->isInitialized()) // TODO: Add timing condition to fallback to GPS if WAISlam too slow
    {
        //Utils::log("AreaTrackingView", "frame res: %d, %d", frame->imgManip.cols, frame->imgManip.rows);
        //the intrinsics may change dynamically on focus changes (e.g. on iOS)
        if (!frame->intrinsics.empty())
        {
            _waiSlam->changeIntrinsic(_camera->scaledCameraMat(), _camera->calibration()->distortion());
            updateSceneCameraFov();
        }

        _waiSlam->update(frame->imgManip);
        if (WAI::TrackingState_TrackingOK == _waiSlam->getTrackingState())
        {
            SLMat4f pose      = convertWAISlamToSLMat(_waiSlam->getPose());
            _transitionMatrix = gpsPose;
            _transitionMatrix.invert();
            _transitionMatrix = pose * _transitionMatrix;
            _waiScene.camera->om(pose);
            _gui.showInfoText("waislam tracking");
        }
        else
        {
            _waiScene.camera->om(_transitionMatrix * gpsPose);
            cv::Mat camExtrinsic = convertCameraPoseToWaiCamExtrinisc(gpsPose);
            _waiSlam->setCamExrinsicGuess(camExtrinsic);
            _gui.showInfoText("gps + sensors tracking");
        }
    }
    else
    {
        _gui.showInfoText("waislam not initizalied");
    }

    return true;
}

bool AreaTrackingView::updateGPS(SENSFramePtr& frame)
{
    if (_camera)
        frame = _camera->latestFrame();
    //use gps and orientation sensor for camera position and orientation
    //(even if there is no gps, devLoc gives us a guess of the current home position)
    SLMat4f gpsPose = calcCameraPoseGpsOrientationBased();
    applyFingerCorrection(gpsPose);

    //SLMat4f gpsPoseCorr = _cameraFingerCorr.getCorrectionMat(focalLength) * gpsPose;
    _waiScene.camera->om(gpsPose);
    _gui.showInfoText("GPS + Sensors positioning");

    return false;
}

void AreaTrackingView::applyFingerCorrection(SLMat4f& camPose)
{
    float   focalLength = scrH() / (2 * tan(0.5f * this->camera()->fovV() * DEG2RAD));
    float   rotAngRAD   = _cameraFingerCorr.getRotAngleRAD(focalLength);
    SLMat4f rot;
    rot.translate(camPose.translation());
    rot.rotate(rotAngRAD * Utils::RAD2DEG, SLVec3f(0, 1, 0));
    rot.translate(-camPose.translation()); //this is multiplied first

    camPose = rot * camPose;
}

bool AreaTrackingView::update()
{
    try
    {
        if (_noInitException) //if there was not exception during initArea
        {
            SENSFramePtr frame = nullptr;

            bool isTracking = false;
            if (_config.useWAISlam && _config.useARCore && _arcore->isAvailable())
                isTracking = updateGPSWAISlamARCore(frame);
            else if (_config.useARCore && _arcore->isAvailable())
                isTracking = updateGPSARCore(frame);
            else if (_config.useWAISlam)
                isTracking = updateWAISlamGPS(frame); //isTracking = updateGPSWAISlam(frame);
            else                                      //fall back to orientation sensor and gps if available
                isTracking = updateGPS(frame);

            if (_asyncLoader && _asyncLoader->isReady())
            {
                if (!_asyncLoader->hasError())
                {
                    _voc              = _asyncLoader->voc();
                    cv::Mat mapNodeOm = _asyncLoader->mapNodeOm();
                    initWaiSlam(mapNodeOm, _asyncLoader->moveWaiMap());
                    _asyncLoader.reset();
                }
                else
                {
                    std::runtime_error exception(_asyncLoader->getErrorMsg());
                    _asyncLoader.reset();
                    throw exception;
                }

                if (_config.enableUserGuidance)
                    _userGuidance.dataIsLoading(false);
            }

            //switch between userguidance scene and tracking scene depending on tracking state
            VideoBackgroundCamera* currentCamera;
            if (isTracking || !_config.enableUserGuidance)
            {
                if (this->s() != &_waiScene)
                {
                    this->scene(&_waiScene);
                    this->camera(_waiScene.camera);
                }
                currentCamera = _waiScene.camera;
            }
            else
            {
                if (this->s() != &_userGuidanceScene)
                {
                    this->scene(&_userGuidanceScene);
                    this->camera(_userGuidanceScene.camera);
                }
                currentCamera = _userGuidanceScene.camera;
            }

            //update visualization
            if (frame)
            {
                //decorate video image and update scene
                //if (_waiSlam)
                //    updateTrackingVisualization(isTracking, *frame.get());

                //set video image camera background
                updateVideoImage(*frame.get(), currentCamera);
            }

            //update user guidance
            if (_config.enableUserGuidance)
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

    if (_config.developerMode && _config.simulatorMode)
    {
        if (_simHelper)
            _simHelper.reset();
        _simHelper = std::make_unique<SENSSimHelper>(_gps,
                                                     _orientation,
                                                     _camera->cameraRef(),
                                                     _deviceData.writableDir() + "SENSSimData",
                                                     std::bind(&AreaTrackingView::onCameraParamsChanged, this));
    }

    _cameraFingerCorr.clear();
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
    if (_arcore)
        _arcore->reset();

    if (_simHelper)
        _simHelper.reset();
}

SLbool AreaTrackingView::onMouseDown(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod)
{
    SLbool ret = SLSceneView::onMouseDown(button, scrX, scrY, mod);

    //Utils::log("AreaTrackingView", "doNotDispatchMouse: %s", _gui.doNotDispatchMouse() ? "true" : "false");
    _gui.mouseDown(button, _gui.doNotDispatchMouse());

    if (button == MB_left && !_gui.doNotDispatchMouse())
        _cameraFingerCorr.onMouseDown(scrX, scrY);

    return ret;
}

SLbool AreaTrackingView::onMouseUp(SLMouseButton button, SLint scrX, SLint scrY, SLKey mod)
{
    SLbool ret = SLSceneView::onMouseUp(button, scrX, scrY, mod);

    _gui.mouseUp(button, _gui.doNotDispatchMouse());
    if (button == MB_left)
    {
        _cameraFingerCorr.onMouseUp(scrX, scrY);
    }

    return ret;
}

SLbool AreaTrackingView::onMouseMove(SLint x, SLint y)
{
    SLbool ret = SLSceneView::onMouseMove(x, y);
    _gui.mouseMove(_gui.doNotDispatchMouse());

    _cameraFingerCorr.onMouseMove(x, y);

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
    if (_gps && _config.useGps)
    {
        auto loc = _gps->getLocation();
        //Update deviceLocation: this updates current enu position
        _devLoc.onLocationLatLonAlt(loc.latitudeDEG, loc.longitudeDEG, loc.altitudeM, loc.accuracyM);
    }

    auto     sensQuat = _orientation->getOrientation();
    SLQuat4f slQuat(sensQuat.quatX, sensQuat.quatY, sensQuat.quatZ, sensQuat.quatW);
    SLMat3f  rotMat = slQuat.toMat3();

    SLMat4f camPose;
    {
        SLMat3f sRc;
        sRc.rotation(-90, 0, 0, 1);

        //sensor rotation w.r.t. east-north-up
        SLMat3f enuRs;
        enuRs.setMatrix(rotMat);

        //enu rotation w.r.t. world
        SLMat3f wRenu;
        wRenu.rotation(-90, 1, 0, 0);

        //combiniation of partial rotations to orientation of camera w.r.t world
        SLMat3f wRc = wRenu * enuRs * sRc;
        camPose.setRotation(wRc);
    }

    //The scene must have a global reference position
    if (_devLoc.hasOrigin())
    {
        // Direction vector from camera to world origin
        SLVec3d wtc = _devLoc.locENU() - _devLoc.originENU();

        // Reset to default if device is too far away
        if (wtc.length() > _devLoc.locMaxDistanceM() || !_config.useGps)
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
    _asyncLoader.reset();
    _asyncLoader = std::make_unique<MapLoader>(area.vocLayer, vocFileName, _deviceData.erlebARDir(), area.slamMapFileName);
    _asyncLoader->start();
    if (_config.enableUserGuidance)
        _userGuidance.dataIsLoading(true);

#else
    if (Utils::fileExists(vocFileName))
    {
        Utils::log("AreaTrackingView", "loading voc file from: %s", vocFileName.c_str());
        _voc = std::make_unique<WAIOrbVocabulary>();
        _voc->loadFromFile(vocFileName);

        //try to load map
        cv::Mat                 mapNodeOm;
        std::unique_ptr<WAIMap> waiMap = tryLoadMap(_deviceData.erlebARDir(), area.slamMapFileName, _voc.get(), mapNodeOm);
        initWaiSlam(mapNodeOm, std::move(waiMap));
    }
    else
    {
        std::stringstream ss;
        ss << "AreaTrackingView initArea: vocabulary file does not exist: " << vocFileName;
        throw std::runtime_error(ss.str());
    }
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
      _voc.get(),
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
            Utils::log("AreaTrackingView", "loading map file successful: %s", mapLoadingSuccess ? "true" : "false");
        }
    }

    return waiMap;
}

void AreaTrackingView::initDeviceLocation(const ErlebAR::Location& location, const ErlebAR::Area& area)
{
    //reset everything to default
    _devLoc.init();

    _devLoc.locMaxDistanceM(1000.0f);
    _devLoc.improveOrigin(false);
    _devLoc.useOriginAltitude(false);
    _devLoc.cameraHeightM(1.6f);
    // Let the sun be rotated by time and location
    if (_waiScene.sunLight)
        _devLoc.sunLightNode(_waiScene.sunLight);

    _devLoc.originLatLonAlt(area.modelOrigin.x, area.modelOrigin.y, area.modelOrigin.z); // Model origin
    _devLoc.defaultLatLonAlt(area.llaPos.x, area.llaPos.y, area.llaPos.z + _devLoc.cameraHeightM());
    //ATTENTION: call this after originLatLonAlt and defaultLatLonAlt setters. Otherwise alititude will be overwritten!!
    if (_config.useGps && !location.geoTiffFileName.empty())
    {
        std::string geoTiffFileName = _deviceData.erlebARDir() + location.geoTiffFileName;
        if (Utils::fileExists(geoTiffFileName))
            _devLoc.loadGeoTiff(geoTiffFileName);
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
            //bars top and bottom: estimate vertical fovV from cameras horizontal field of view and screen aspect ratio
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
            _camera->configure(SENSCameraFacing::BACK, TARGET_WIDTH, TARGET_HEIGHT, trackImgSize.width, trackImgSize.height, false, false, true);
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
    if (_config.developerMode)
        _waiSlam->drawInfo(frame.imgBGR, frame.scaleToManip, true, false, true);

    //update map point visualization
    if (_config.developerMode)
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
