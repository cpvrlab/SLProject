#include <WAIApp.h>

#include <SLKeyframeCamera.h>
#include <Utils.h>
#include <AverageTiming.h>

#include <WAIMapStorage.h>
#include <WAICalibration.h>
#include <AppWAIScene.h>

//#include <AppDemoGuiInfosDialog.h>
//#include <AppDemoGuiAbout.h>
//#include <AppDemoGuiInfosFrameworks.h>
//#include <AppDemoGuiInfosMapNodeTransform.h>
//#include <AppDemoGuiInfosScene.h>
//#include <AppDemoGuiInfosSensors.h>
//#include <AppDemoGuiInfosTracking.h>
//#include <AppDemoGuiProperties.h>
//#include <AppDemoGuiSceneGraph.h>
//#include <AppDemoGuiStatsDebugTiming.h>
//#include <AppDemoGuiStatsTiming.h>
//#include <AppDemoGuiStatsVideo.h>
//#include <AppDemoGuiTrackedMapping.h>
//#include <AppDemoGuiTransform.h>
//#include <AppDemoGuiUIPrefs.h>
//#include <AppDemoGuiVideoControls.h>
//#include <AppDemoGuiVideoStorage.h>
//#include <AppDemoGuiSlamLoad.h>
//#include <AppDemoGuiTestOpen.h>
//#include <AppDemoGuiTestWrite.h>
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

    //Utils::initFileLog(_dirs.logFileDir, true);

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
    std::lock_guard<std::mutex> lock(_cameraSetMutex);
    _camera = camera;
}

SENSCamera* WAIApp::getCamera()
{
    std::lock_guard<std::mutex> lock(_cameraSetMutex);
    return _camera;
}

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
            if (_startUpState && _startUpState->ready())
            {
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
            if (_goBack)
            {
                _goBack  = false;
                _state   = State::SELECTION;
                _appMode = AppMode::NONE;
                _selectionState->reset();
            }
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
                    if (!_selectionState)
                    {
                        _selectionState = new SelectionState(_inputManager,
                                                             _deviceData->scrWidth(),
                                                             _deviceData->scrHeight(),
                                                             _deviceData->dpi(),
                                                             _deviceData->fontDir(),
                                                             _deviceData->dirs().writableDir);
                        _selectionState->start();
                    }
                }

                _startUpState = new StartUpState(_inputManager,
                                                 _deviceData->scrWidth(),
                                                 _deviceData->scrHeight(),
                                                 _deviceData->dpi(),
                                                 _deviceData->dirs().writableDir);
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

            if (_startFromStartUp && getCamera() && _camera->started())
            {
                _startFromStartUp = false;
                _appMode          = _selectionState->getSelection();

                if (_appMode == AppMode::TEST)
                {
                    if (!_testState)
                    {
                        _testState = new TestState(_inputManager,
                                                   getCamera(),
                                                   _deviceData->scrWidth(),
                                                   _deviceData->scrHeight(),
                                                   _deviceData->dpi(),
                                                   _deviceData->fontDir(),
                                                   _deviceData->dirs().writableDir,
                                                   _deviceData->dirs().vocabularyDir,
                                                   _deviceData->calibDir(),
                                                   _deviceData->videoDir(),
                                                   std::bind(&WAIApp::goBack, this));
                        _testState->start();
                    }
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
            doUpdate = _testState->update();
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
    terminate();
}

void WAIApp::terminate()
{
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

std::string WAIApp::name()
{
    return _name;
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
