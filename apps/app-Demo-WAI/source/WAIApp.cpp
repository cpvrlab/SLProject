#include <WAIApp.h>

#include <SLKeyframeCamera.h>
#include <Utils.h>
#include <AverageTiming.h>

#include <WAIModeOrbSlam2.h>
#include <WAIMapStorage.h>
#include <WAICalibration.h>
#include <AppWAIScene.h>

#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiAbout.h>
#include <AppDemoGuiInfosFrameworks.h>
#include <AppDemoGuiInfosMapNodeTransform.h>
#include <AppDemoGuiInfosScene.h>
#include <AppDemoGuiInfosSensors.h>
#include <AppDemoGuiInfosTracking.h>
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
#include <AppDemoGuiError.h>

#include <DeviceData.h>
#include <AppWAISlamParamHelper.h>
#include <FtpUtils.h>
#include <GUIPreferences.h>

#include <GLSLextractor.h>
#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLQuat4.h>
#include <SLPolyline.h>
#include <opencv2/imgproc/imgproc.hpp>

//move
#include <SLAssimpImporter.h>

//basic information
//-the sceen has a fixed size on android and also a fixed aspect ratio on desktop to simulate android behaviour on desktop
//-the viewport gets adjusted according to the target video aspect ratio (for live video WAIApp::liveVideoTargetWidth and WAIApp::_liveVideoTargetHeight are used and for video file the video frame size is used)
//-the live video gets cropped according to the viewport aspect ratio on android and according to WAIApp::videoFrameWdivH on desktop (which is also used to define the viewport aspect ratio)
//-the calibration gets adjusted according to the video (live and file)
//-the live video gets cropped to the aspect ratio that is defined by the transferred values in load(..) and assigned to liveVideoTargetWidth and liveVideoTargetHeight

//-----------------------------------------------------------------------------
WAIApp::WAIApp()
  : SLInputEventInterface(_inputManager),
    _name("WAI Demo App")
{
}
//-----------------------------------------------------------------------------
WAIApp::~WAIApp()
{
    //todo: destructor is not called on android (at the right position)
    if (_videoWriter)
        delete _videoWriter;
    if (_mode)
        delete _mode;
}

//-----------------------------------------------------------------------------
int WAIApp::load(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi, AppDirectories directories)
{
    _dirs = directories;
    //todo ghm1: load sets state to startup: we can visualize
    //videoDir  = _dirs.writableDir + "erleb-AR/locations/";
    //_calibDir = _dirs.writableDir + "calibrations/";
    //mapDir    = _dirs.writableDir + "maps/";

    _deviceData = std::make_unique<DeviceData>(scrWidth, scrHeight, scr2fbX, scr2fbY, dpi, directories);
    // setup magic paths
    SLGLProgram::defaultPath      = _dirs.slDataRoot + "/shaders/";
    SLGLTexture::defaultPath      = _dirs.slDataRoot + "/images/textures/";
    SLGLTexture::defaultPathFonts = _dirs.slDataRoot + "/images/fonts/";
    SLAssimpImporter::defaultPath = _dirs.slDataRoot + "/models/";

    _startFromIdle = true;
    return 0;
    //if (_loaded) return _sv->index();

    ////Utils::initFileLog(_dirs.logFileDir, true);

    //Utils::log("WAInative", "loading");
    //_dirs = directories;

    //videoDir = _dirs.writableDir + "erleb-AR/locations/";
    //_calibDir = _dirs.writableDir + "calibrations/";
    //mapDir    = _dirs.writableDir + "maps/";

    ////init scene as soon as possible to allow visualization of error msgs
    //int svIndex = initSLProject(scrWidth, scrHeight, scr2fbX, scr2fbY, dpi);

    //setupDefaultErlebARDirTo(_dirs.writableDir);

    //_loaded = true;

    //return svIndex;
}

void WAIApp::setCamera(SENSCamera* camera)
{
    _camera = camera;
    if (_sv)
        _sv->setViewportFromRatio(SLVec2i(_camera->getFrameSize().width, _camera->getFrameSize().height), SLViewportAlign::VA_center, true);
}

void WAIApp::loadSlam()
{
    if (_currentSlamParams.load(_dirs.writableDir + "SlamParams.json"))
    {
        loadWAISceneView(_currentSlamParams.location, _currentSlamParams.area);
        startOrbSlam(_currentSlamParams);
        _guiSlamLoad->setSlamParams(_currentSlamParams);
        _gui->uiPrefs->showSlamLoad = false;
    }
    else
    {
        _gui->uiPrefs->showSlamLoad = true;
    }
}

SENSFramePtr WAIApp::getVideoFrame()
{
    SENSFramePtr frame;

    if (_videoFileStream)
    {
        while (_videoCursorMoveIndex < 0)
        {
            frame = _videoFileStream->grabPreviousFrame();
            if (frame)
                updateTracking(frame);

            _videoCursorMoveIndex++;
        }

        while (_videoCursorMoveIndex > 0)
        {
            frame = _videoFileStream->grabNextFrame();
            if (frame)
                updateTracking(frame);

            _videoCursorMoveIndex--;
        }

        if (!_pauseVideo)
        {
            frame = _videoFileStream->grabNextFrame();
        }
    }
    else
    {
        Utils::log("WAI WARN", "WAIApp::getVideoFrame: No active video stream available!");
    }

    return std::move(frame);
}

SENSFramePtr WAIApp::getCameraFrame()
{
    SENSFramePtr frame;

    if (_camera)
        frame = _camera->getLatestFrame();
    else
        Utils::log("WAI WARN", "WAIApp::updateVideoOrCamera: No active camera available!");

    return std::move(frame);
}

//SENSFramePtr WAIApp::updateVideoOrCamera()
//{
//    SENSFramePtr frame;
//
//    if (_videoFileStream)
//    {
//        while (_videoCursorMoveIndex < 0)
//        {
//            frame = _videoFileStream->grabPreviousFrame();
//            if (frame)
//                updateTracking(frame);
//
//            _videoCursorMoveIndex++;
//        }
//
//        while (_videoCursorMoveIndex > 0)
//        {
//            frame = _videoFileStream->grabNextFrame();
//            if (frame)
//                updateTracking(frame);
//
//            _videoCursorMoveIndex--;
//        }
//
//        if (!_pauseVideo)
//        {
//            frame = _videoFileStream->grabNextFrame();
//        }
//    }
//
//    else if (_camera)
//    {
//        frame = _camera->getLatestFrame();
//    }
//    else
//    {
//        Utils::log("WAI WARN", "WAIApp::updateVideoOrCamera: No active camera available!");
//    }
//
//    return std::move(frame);
//}

void WAIApp::reset()
{
    _startFromIdle    = false;
    _startFromStartUp = false;
}

void WAIApp::checkStateTransition()
{
    switch (_state)
    {
        case State::IDLE: {
            if (_appMode == AppMode::NONE)
            {
                //start selection state
                if (_selectionState && _selectionState->started())
                    _state = State::SELECTION;
            }
            else
            {
                //directly go to start up
                if (_startUpState && _startUpState->started())
                    _state = State::START_UP;
            }
            break;
        }
        case State::SELECTION: {

            if (_appMode != AppMode::NONE && _startUpState->started())
            {
                _state            = State::START_UP;
                _startFromStartUp = true;
            }
            break;
        }
        case State::START_UP: {
            if (_appMode == AppMode::CAMERA_TEST)
            {
                if (_cameraTestState && _cameraTestState->started())
                {
                    _state = State::CAMERA_TEST;
                }
            }
            else if (_appMode == AppMode::TEST)
            {
                if (_testState && _testState->started())
                {
                    _state = State::TEST;
                }
            }
            else //(_appMode > AppMode::TEST)
            {
                if (_locationMapState && _areaTrackingState && _locationMapState->started() && _areaTrackingState->started())
                {
                    _state = State::LOCATION_MAP;
                }
            }
            break;
        }
        case State::LOCATION_MAP: {
            //if (_mapState.ready() && _slamState.started())
            {
                //  _state = State::MAP_SCENE
            }
            break;
        }
        case State::AREA_TRACKING: {
            break;
        }
        case State::TEST: {
            break;
        }
        case State::CAMERA_TEST: {
            break;
        }
        case State::TUTORIAL: {
            break;
        }
    }
}

bool WAIApp::updateState()
{
    bool doUpdate = false;
    switch (_state)
    {
        case State::IDLE: {
            if (_startFromIdle)
            {
                _startFromIdle = false;
                if (_appMode == AppMode::NONE)
                {
                    //select AppMode
                    //(start up can be done as soon we have access to resouces)
                    _selectionState = new SelectionState(_inputManager,
                                                         _deviceData->scrWidth(),
                                                         _deviceData->scrHeight(),
                                                         _deviceData->dpi(),
                                                         _deviceData->fontDir(),
                                                         _deviceData->dirs().writableDir);
                    _selectionState->start();
                }

                _startUpState = new StartUpState;
                _startUpState->start();
            }
            break;
        }
        case State::SELECTION: {
            doUpdate = _selectionState->update();
            if (_selectionState->ready())
            {
                _appMode = _selectionState->getSelection();
            }
            break;
        }
        case State::START_UP: {
            //show loading screen
            doUpdate = _startUpState->update();

            if (_startFromStartUp)
            {
                _startFromStartUp = false;
                _appMode          = _selectionState->getSelection();

                if (_appMode == AppMode::TEST)
                {
                    _testState = new TestState;
                    _testState->start();
                }
                else if (_appMode == AppMode::CAMERA_TEST)
                {
                    _cameraTestState = new CameraTestState;
                    _cameraTestState->start();
                }
                else
                {
                    if (_appMode == AppMode::AUGST)
                        _locationMapState = new AugstMapState;
                    else if (_appMode == AppMode::AVANCHES)
                        _locationMapState = new AvanchesMapState;
                    else if (_appMode == AppMode::BIEL)
                        _locationMapState = new BielMapState;
                    else if (_appMode == AppMode::CHRISTOFFELTOWER)
                        _locationMapState = new ChristoffelMapState;

                    _locationMapState->start();
                }
            }
            break;
        }
        case State::LOCATION_MAP: {

            doUpdate = _locationMapState->update();
            if (_locationMapState->ready())
            {
                _area             = _locationMapState->getTargetArea();
                _switchToTracking = true;
            }
            break;
        }
        case State::AREA_TRACKING: {
            doUpdate = _areaTrackingState->update();

            if (_areaTrackingState->ready())
            {
                //todo:
            }

            break;
        }
        case State::TEST: {
            break;
        }
        case State::CAMERA_TEST: {
            break;
        }
        case State::TUTORIAL: {
            break;
        }
    }

    return doUpdate;
}

bool WAIApp::update()
{
    bool doUpdate = false;

    try
    {
        checkStateTransition();
        doUpdate = updateState();
    }
    catch (std::exception& e)
    {
        Utils::log("WAIApp", "Std exception catched in update() %s", e.what());
    }
    catch (...)
    {
        Utils::log("WAIApp", "Unknown exception catched in update()");
    }

    return doUpdate;
}

void WAIApp::close()
{
    _camera = nullptr;
    if (_mode)
        _currentSlamParams.save(_dirs.writableDir + "SlamParams.json");
    terminate();
}

void WAIApp::terminate()
{
    // Deletes all remaining sceneviews the current scene instance
    if (_waiScene)
    {
        delete _waiScene;
        _waiScene = nullptr;
    }

    if (_selectionState)
    {
        delete _selectionState;
        _selectionState = nullptr;
    }
    if (_startUpState)
    {
        delete _startUpState;
        _startUpState = nullptr;
    }
    if (_areaTrackingState)
    {
        delete _areaTrackingState;
        _areaTrackingState = nullptr;
    }
    if (_cameraTestState)
    {
        delete _cameraTestState;
        _cameraTestState = nullptr;
    }
    if (_locationMapState)
    {
        delete _locationMapState;
        _locationMapState = nullptr;
    }
    if (_testState)
    {
        delete _testState;
        _testState = nullptr;
    }
    if (_tutorialState)
    {
        delete _tutorialState;
        _tutorialState = nullptr;
    }
}

/*
videoFile: path to a video or empty if live video should be used
calibrationFile: path to a calibration or empty if calibration should be searched automatically
mapFile: path to a map or empty if no map should be used
*/
void WAIApp::startOrbSlam(SlamParams slamParams)
{
    _errorDial->setErrorMsg("");
    _gui->uiPrefs->showError = false;
    _lastFrameIdx            = 0;
    _doubleBufferedOutput    = false;
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
        slamParams.calibrationFile = _deviceData->calibDir() + calibrationFileName;
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
    if (!_calibration.load(_deviceData->calibDir(), Utils::getFileName(slamParams.calibrationFile), false))
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
    _waiScene->updateCameraIntrinsics(_calibration.cameraFovVDeg(), _calibration.cameraMatUndistorted());
    //  _waiScene->cameraNode->fov(_calibration.cameraFovVDeg());
    //// Set camera intrinsics for scene camera frustum. (used in projection->intrinsics mode)
    //cv::Mat scMat = _calibration.cameraMatUndistorted();
    //std::cout << "scMat: " << scMat << std::endl;
    //_waiScene->cameraNode->intrinsics(scMat.at<double>(0, 0),
    //                                  scMat.at<double>(1, 1),
    //                                  scMat.at<double>(0, 2),
    //                                  scMat.at<double>(1, 2));

    ////enable projection -> intrinsics mode
    //_waiScene->cameraNode->projection(P_monoIntrinsic);

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
                                                        _waiScene->mapNode,
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

    _sv->setViewportFromRatio(SLVec2i(_videoFrameSize.width, _videoFrameSize.height), SLViewportAlign::VA_center, true);
    //_resizeWindow = true;
    _undistortedLastFrame[0] = cv::Mat(_videoFrameSize.height, _videoFrameSize.width, CV_8UC3);
    _undistortedLastFrame[1] = cv::Mat(_videoFrameSize.height, _videoFrameSize.width, CV_8UC3);
}

void WAIApp::showErrorMsg(std::string msg)
{
    assert(_errorDial && "errorDial is not initialized");

    _errorDial->setErrorMsg(msg);
    _gui->uiPrefs->showError = true;
}

std::string WAIApp::name()
{
    return _name;
}

bool WAIApp::updateTracking(SENSFramePtr frame)
{
    bool iKnowWhereIAm = false;

    if (_videoWriter && _videoWriter->isOpened())
    {
        _videoWriter->write(frame->imgRGB);
    }

    iKnowWhereIAm = _mode->update(frame->imgGray);

    //if (_gpsDataStream.is_open())
    //{
    //    if (SLApplication::devLoc.isUsed())
    //    {
    //        SLVec3d v = SLApplication::devLoc.locLLA();
    //        _gpsDataStream << SLApplication::devLoc.locAccuracyM();
    //        _gpsDataStream << std::to_string(v.x) + " " + std::to_string(v.y) + " " + std::to_string(v.z);
    //        _gpsDataStream << std::to_string(SLApplication::devRot.yawRAD());
    //        _gpsDataStream << std::to_string(SLApplication::devRot.pitchRAD());
    //        _gpsDataStream << std::to_string(SLApplication::devRot.rollRAD());
    //    }
    //}

    return iKnowWhereIAm;
}

//int WAIApp::initSLProject(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi)
//{
//    if (!_waiScene)
//    {
//        // Default paths for all loaded resources
//        SLGLProgram::defaultPath      = _dirs.slDataRoot + "/shaders/";
//        SLGLTexture::defaultPath      = _dirs.slDataRoot + "/images/textures/";
//        SLGLTexture::defaultPathFonts = _dirs.slDataRoot + "/images/fonts/";
//        SLAssimpImporter::defaultPath = _dirs.slDataRoot + "/models/";
//
//        _waiScene = new AppWAIScene(_name, _inputManager);
//
//        int screenWidth  = (int)(scrWidth * scr2fbX);
//        int screenHeight = (int)(scrHeight * scr2fbY);
//
//        setupGUI(_name, _dirs.writableDir, dpi);
//
//        _sv = new SLSceneView(_waiScene, dpi);
//        _sv->init("SceneView",
//                  screenWidth,
//                  screenHeight,
//                  nullptr,
//                  nullptr,
//                  _gui.get(),
//                  _dirs.writableDir);
//
//        //loadWAISceneView("default", "default");
//    }
//
//    return (SLint)_sv->index();
//}

void WAIApp::loadWAISceneView(std::string location, std::string area)
{
    _waiScene->rebuild(location, area);

    _sv->doWaitOnIdle(false);
    _sv->camera(_waiScene->cameraNode);
    _sv->onInitialize();
    _sv->setViewportFromRatio(SLVec2i(_camera->getFrameSize().width, _camera->getFrameSize().height), SLViewportAlign::VA_center, true);
}

void WAIApp::setupGUI(std::string appName, std::string configDir, int dotsPerInch)
{
    _gui = std::make_unique<AppDemoWaiGui>(_name, _dirs.writableDir, dotsPerInch, _dirs.slDataRoot + "/images/fonts/");

    _gui->addInfoDialog(std::make_shared<AppDemoGuiInfosFrameworks>("frameworks", &_gui->uiPrefs->showInfosFrameworks));
    _gui->addInfoDialog(std::make_shared<AppDemoGuiInfosMapNodeTransform>("map node",
                                                                          &_gui->uiPrefs->showInfosMapNodeTransform,
                                                                          &_eventQueue));

    _gui->addInfoDialog(std::make_shared<AppDemoGuiInfosScene>("scene", &_gui->uiPrefs->showInfosScene));
    _gui->addInfoDialog(std::make_shared<AppDemoGuiInfosSensors>("sensors", &_gui->uiPrefs->showInfosSensors));
    _gui->addInfoDialog(std::make_shared<AppDemoGuiInfosTracking>("tracking", *_gui->uiPrefs.get(), *this));

    _guiSlamLoad = std::make_shared<AppDemoGuiSlamLoad>("slam load",
                                                        &_eventQueue,
                                                        _dirs.writableDir + "erleb-AR/locations/",
                                                        _dirs.writableDir + "calibrations/",
                                                        _dirs.vocabularyDir,
                                                        _featureExtractorFactory.getExtractorIdToNames(),
                                                        &_gui->uiPrefs->showSlamLoad);
    _gui->addInfoDialog(_guiSlamLoad);

    _gui->addInfoDialog(std::make_shared<AppDemoGuiProperties>("properties", &_gui->uiPrefs->showProperties));
    _gui->addInfoDialog(std::make_shared<AppDemoGuiSceneGraph>("scene graph", &_gui->uiPrefs->showSceneGraph));
    _gui->addInfoDialog(std::make_shared<AppDemoGuiStatsDebugTiming>("debug timing", &_gui->uiPrefs->showStatsDebugTiming));
    _gui->addInfoDialog(std::make_shared<AppDemoGuiStatsTiming>("timing", &_gui->uiPrefs->showStatsTiming));
    _gui->addInfoDialog(std::make_shared<AppDemoGuiStatsVideo>("video", &_gui->uiPrefs->showStatsVideo, *this));
    _gui->addInfoDialog(std::make_shared<AppDemoGuiTrackedMapping>("tracked mapping", &_gui->uiPrefs->showTrackedMapping, *this));

    _gui->addInfoDialog(std::make_shared<AppDemoGuiTransform>("transform", &_gui->uiPrefs->showTransform));
    _gui->addInfoDialog(std::make_shared<AppDemoGuiUIPrefs>("prefs", _gui->uiPrefs.get(), &_gui->uiPrefs->showUIPrefs));

    _gui->addInfoDialog(std::make_shared<AppDemoGuiVideoStorage>("video/gps storage", &_gui->uiPrefs->showVideoStorage, &_eventQueue, *this));
    _gui->addInfoDialog(std::make_shared<AppDemoGuiVideoControls>("video load", &_gui->uiPrefs->showVideoControls, &_eventQueue, *this));

    _errorDial = std::make_shared<AppDemoGuiError>("Error", &_gui->uiPrefs->showError);
    _gui->addInfoDialog(_errorDial);
}

void WAIApp::setupDefaultErlebARDirTo(std::string dir)
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

void WAIApp::downloadCalibrationFilesTo(std::string dir)
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

void WAIApp::updateTrackingVisualization(const bool iKnowWhereIAm, cv::Mat& imgRGB)
{
    //undistort image and copy image to video texture
    _mode->drawInfo(imgRGB, true, _gui->uiPrefs->showKeyPoints, _gui->uiPrefs->showKeyPointsMatched);

    if (_calibration.state() == CS_calibrated && _showUndistorted)
        _calibration.remap(imgRGB, _undistortedLastFrame[_lastFrameIdx]);
    else
        _undistortedLastFrame[_lastFrameIdx] = imgRGB;

    if (_doubleBufferedOutput)
        _lastFrameIdx = (_lastFrameIdx + 1) % 2;

    _waiScene->updateVideoImage(_undistortedLastFrame[_lastFrameIdx],
                                CVImage::cv2glPixelFormat(_undistortedLastFrame[_lastFrameIdx].type()));

    //update map point visualization
    if (_gui->uiPrefs->showMapPC)
    {
        _waiScene->renderMapPoints(_mode->getMapPoints());
        //todo: fix? mode has no member getMarmerCornerMapPoints
        //_waiScene->renderMarkerCornerMapPoints(_mode->getMarkerCornerMapPoints());
    }
    else
    {
        _waiScene->removeMapPoints();
        _waiScene->removeMarkerCornerMapPoints();
    }

    //update visualization of local map points (when WAI pose is valid)
    if (_gui->uiPrefs->showLocalMapPC && iKnowWhereIAm)
        _waiScene->renderLocalMapPoints(_mode->getLocalMapPoints());
    else
        _waiScene->removeLocalMapPoints();

    //update visualization of matched map points (when WAI pose is valid)
    if (_gui->uiPrefs->showMatchesPC && iKnowWhereIAm)
        _waiScene->renderMatchedMapPoints(_mode->getMatchedMapPoints(_mode->getLastFrame()));
    else
        _waiScene->removeMatchedMapPoints();

    //update keyframe visualization
    if (_gui->uiPrefs->showKeyFrames)
        _waiScene->renderKeyframes(_mode->getKeyFrames());
    else
        _waiScene->removeKeyframes();

    //update pose graph visualization
    _waiScene->renderGraphs(_mode->getKeyFrames(),
                            _gui->uiPrefs->minNumOfCovisibles,
                            _gui->uiPrefs->showCovisibilityGraph,
                            _gui->uiPrefs->showSpanningTree,
                            _gui->uiPrefs->showLoopEdges);
}

void WAIApp::saveMap(std::string location,
                     std::string area,
                     std::string marker)
{
    _mode->requestStateIdle();

    std::string slamRootDir = _dirs.writableDir + "erleb-AR/locations/";
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
            //_waiScene->mapNode->om(WAIMapStorage::convertToSLMat(nodeTransform));
            if (!WAIMapStorage::saveMap(_mode->getMap(),
                                        _waiScene->mapNode,
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
                                    _waiScene->mapNode,
                                    mapDir + filename,
                                    imgDir))
        {
            showErrorMsg("Failed to save map " + mapDir + filename);
        }
    }

    _mode->resume();
}

void WAIApp::saveVideo(std::string filename)
{
    std::string infoDir  = _deviceData->videoDir() + "info/";
    std::string infoPath = infoDir + filename;
    std::string path     = _deviceData->videoDir() + filename;

    if (!Utils::dirExists(_deviceData->videoDir()))
    {
        Utils::makeDir(_deviceData->videoDir());
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

//void WAIApp::saveGPSData(std::string videofile)
//{
//    if (_gpsDataStream.is_open())
//        _gpsDataStream.close();
//
//    std::string filename = Utils::getFileNameWOExt(videofile) + ".txt";
//    std::string path     = videoDir + filename;
//    _gpsDataStream.open(path);
//}

void WAIApp::transformMapNode(SLTransformSpace tSpace,
                              SLVec3f          rotation,
                              SLVec3f          translation,
                              float            scale)
{
    _waiScene->mapNode->rotate(rotation.x, 1, 0, 0, tSpace);
    _waiScene->mapNode->rotate(rotation.y, 0, 1, 0, tSpace);
    _waiScene->mapNode->rotate(rotation.z, 0, 0, 1, tSpace);
    _waiScene->mapNode->translate(translation.x, 0, 0, tSpace);
    _waiScene->mapNode->translate(0, translation.y, 0, tSpace);
    _waiScene->mapNode->translate(0, 0, translation.z, tSpace);
    _waiScene->mapNode->scale(scale);
}

void WAIApp::handleEvents()
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
                downloadCalibrationFilesTo(_deviceData->calibDir());
            }
            break;

            case WAIEventType_AdjustTransparency: {
                WAIEventAdjustTransparency* adjustTransparencyEvent = (WAIEventAdjustTransparency*)event;
                _waiScene->adjustAugmentationTransparency(adjustTransparencyEvent->kt);

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
