#include "ErlebARApp.h"
#include <SLGLProgram.h>
#include <SLGLTexture.h>
#include <SLAssimpImporter.h>
#include <views/SelectionView.h>
#include <views/TestView.h>
#include <views/TestRunnerView.h>
#include <views/StartUpView.h>
#include <views/WelcomeView.h>
#include <views/SettingsView.h>
#include <views/AboutView.h>
#include <views/TutorialView.h>
#include <views/AreaTrackingView.h>
#include <views/LocationMapView.h>
#include <views/AreaInfoView.h>
#include <views/CameraTestView.h>

#include <SLGLProgramManager.h>

#define LOG_ERLEBAR_WARN(...) Utils::log("ErlebARApp", __VA_ARGS__);
#define LOG_ERLEBAR_INFO(...) Utils::log("ErlebARApp", __VA_ARGS__);
#define LOG_ERLEBAR_DEBUG(...) Utils::log("ErlebARApp", __VA_ARGS__);

ErlebARApp::ErlebARApp()
  : sm::StateMachine((unsigned int)StateId::IDLE),
    SLInputEventInterface(_inputManager)
{
    registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::IDLE>((unsigned int)StateId::IDLE);
    registerState<ErlebARApp, InitEventData, &ErlebARApp::INIT>((unsigned int)StateId::INIT);
    registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::WELCOME>((unsigned int)StateId::WELCOME);
    registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::DESTROY>((unsigned int)StateId::DESTROY);
    registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::SELECTION>((unsigned int)StateId::SELECTION);

    registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::START_TEST>((unsigned int)StateId::START_TEST);
    registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::TEST>((unsigned int)StateId::TEST);

    registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::TEST_RUNNER>((unsigned int)StateId::TEST_RUNNER);

    registerState<ErlebARApp, ErlebarEventData, &ErlebARApp::LOCATION_MAP>((unsigned int)StateId::LOCATION_MAP);
    registerState<ErlebARApp, AreaEventData, &ErlebARApp::AREA_INFO>((unsigned int)StateId::AREA_INFO);
    registerState<ErlebARApp, AreaEventData, &ErlebARApp::AREA_TRACKING>((unsigned int)StateId::AREA_TRACKING);
    registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::HOLD_TRACKING>((unsigned int)StateId::HOLD_TRACKING);

    registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::TUTORIAL>((unsigned int)StateId::TUTORIAL);
    registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::SETTINGS>((unsigned int)StateId::SETTINGS);
    registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::ABOUT>((unsigned int)StateId::ABOUT);
    registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::CAMERA_TEST>((unsigned int)StateId::CAMERA_TEST);
}

void ErlebARApp::init(int                scrWidth,
                      int                scrHeight,
                      int                dpi,
                      const std::string& dataDir,
                      const std::string& writableDir,
                      SENSCamera*        camera)
{
    //store camera so we can stop on terminate
    _camera = camera;
    addEvent(new InitEvent("ErlebARApp::init()", scrWidth, scrHeight, dpi, dataDir, writableDir));
}

void ErlebARApp::goBack()
{
    addEvent(new GoBackEvent("ErlebARApp::goBack()"));
}

void ErlebARApp::destroy()
{
    addEvent(new DestroyEvent("ErlebARApp::destroy()"));
}

void ErlebARApp::hold()
{
    addEvent(new HoldEvent("ErlebARApp::hold()"));
}

void ErlebARApp::resume()
{
    addEvent(new ResumeEvent("ErlebARApp::resume()"));
}

std::string ErlebARApp::getPrintableState(unsigned int state)
{
    StateId stateId = (StateId)state;
    switch (stateId)
    {
        case StateId::IDLE:
            return "IDLE";
        case StateId::INIT:
            return "INIT";
        case StateId::WELCOME:
            return "WELCOME";
        case StateId::DESTROY:
            return "DESTROY";
        case StateId::SELECTION:
            return "SELECTION";

        case StateId::START_TEST:
            return "START_TEST";
        case StateId::TEST:
            return "TEST";

        case StateId::TEST_RUNNER:
            return "TEST_RUNNER";

        case StateId::LOCATION_MAP:
            return "LOCATION_MAP";
        case StateId::AREA_INFO:
            return "AREA_INFO";
        case StateId::AREA_TRACKING:
            return "AREA_TRACKING";
        case StateId::HOLD_TRACKING:
            return "HOLD_TRACKING";

        case StateId::TUTORIAL:
            return "TUTORIAL";
        case StateId::ABOUT:
            return "ABOUT";
        case StateId::SETTINGS:
            return "SETTINGS";
        case StateId::CAMERA_TEST:
            return "CAMERA_TEST";
        default: {
            std::stringstream ss;
            ss << "Undefined state or missing string in ErlebARApp::getPrintableState for id: " << state << "!";
            return ss.str();
        }
    }
}

void ErlebARApp::IDLE(const sm::NoEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateExit)
        return;

    if (stateEntry)
        LOG_ERLEBAR_DEBUG("IDLE");
}

void ErlebARApp::INIT(const InitEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateExit)
        return;

    if (stateEntry)
        LOG_ERLEBAR_DEBUG("INIT");

    if (data == nullptr)
        return;

    _dd = std::make_unique<DeviceData>(data->deviceData);

    SLGLProgramManager::init(_dd->shaderDir());
    _resources   = new ErlebAR::Resources(*_dd);
    _imGuiEngine = new ImGuiEngine(_dd->writableDir(), _resources->fonts().atlas());

    //instantiation of views
    _welcomeView = new WelcomeView(_inputManager,
                                   *_resources,
                                   *_imGuiEngine,
                                   *_dd,
                                   "0.12");

    addEvent(new DoneEvent("ErlebARApp::INIT"));
}

void ErlebARApp::WELCOME(const sm::NoEventData* data, const bool stateEntry, const bool stateExit)
{
    static HighResTimer timer;
    if (stateEntry)
    {
        timer.start();
    }
    else
    {
        //init all other views after first rendering of start screen
        if (!_selectionView)
        {
            _selectionView = new SelectionView(*this,
                                               _inputManager,
                                               *_imGuiEngine,
                                               *_resources,
                                               *_dd);
        }

        if (!_testView)
        {
            _testView = new TestView(*this,
                                     _inputManager,
                                     *_imGuiEngine,
                                     *_resources,
                                     _camera,
                                     *_dd);
        }

        if (!_testRunnerView)
        {
            _testRunnerView = new TestRunnerView(*this,
                                                 _inputManager,
                                                 *_imGuiEngine,
                                                 *_resources,
                                                 *_dd);
        }

        if (!_startUpView)
        {
            _startUpView = new StartUpView(_inputManager,
                                           *_dd);
        }

        if (!_aboutView)
        {
            _aboutView = new AboutView(*this,
                                       _inputManager,
                                       *_imGuiEngine,
                                       *_resources,
                                       *_dd);
        }

        if (!_settingsView)
        {
            _settingsView = new SettingsView(*this,
                                             _inputManager,
                                             *_imGuiEngine,
                                             *_resources,
                                             *_dd);
        }

        if (!_tutorialView)
        {
            _tutorialView = new TutorialView(*this,
                                             _inputManager,
                                             *_imGuiEngine,
                                             *_resources,
                                             *_dd);
        }

        if (!_locationMapView)
        {
            _locationMapView = new LocationMapView(*this,
                                                   _inputManager,
                                                   *_imGuiEngine,
                                                   *_resources,
                                                   *_dd);
        }

        if (!_areaInfoView)
        {
            _areaInfoView = new AreaInfoView(*this,
                                             _inputManager,
                                             *_imGuiEngine,
                                             *_resources,
                                             *_dd);
        }

        if (!_areaTrackingView)
        {
            _areaTrackingView = new AreaTrackingView(*this,
                                                     _inputManager,
                                                     *_imGuiEngine,
                                                     *_resources,
                                                     _camera,
                                                     *_dd);
        }
    }

    if (stateExit)
        return;

    _welcomeView->update();

    if (timer.elapsedTimeInSec() > 0.01f)
        addEvent(new DoneEvent("ErlebARApp::WELCOME"));
}

void ErlebARApp::DESTROY(const sm::NoEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateExit)
        return;

    if (_selectionView)
    {
        delete _selectionView;
        _selectionView = nullptr;
    }
    if (_testView)
    {
        delete _testView;
        _testView = nullptr;
    }
    if (_startUpView)
    {
        delete _startUpView;
        _startUpView = nullptr;
    }
    if (_welcomeView)
    {
        delete _welcomeView;
        _welcomeView = nullptr;
    }
    if (_aboutView)
    {
        delete _aboutView;
        _aboutView = nullptr;
    }
    if (_settingsView)
    {
        delete _settingsView;
        _settingsView = nullptr;
    }
    if (_tutorialView)
    {
        delete _tutorialView;
        _tutorialView = nullptr;
    }
    if (_locationMapView)
    {
        delete _locationMapView;
        _locationMapView = nullptr;
    }
    if (_areaInfoView)
    {
        delete _areaInfoView;
        _areaInfoView = nullptr;
    }
    if (_areaTrackingView)
    {
        delete _areaTrackingView;
        _areaTrackingView = nullptr;
    }
    if (_cameraTestView)
    {
        delete _cameraTestView;
        _cameraTestView = nullptr;
    }

    if (_camera)
    {
        if (_camera->started())
        {
            _camera->stop();
        }
    }

    if (_imGuiEngine)
    {
        delete _imGuiEngine;
        _imGuiEngine = nullptr;
    }

    if (_resources)
    {
        delete _resources;
        _resources = nullptr;
    }

    //ATTENTION: if we dont do this we get problems when opening the app the second time
    //(e.g. "The position attribute has no variable location." from SLGLVertexArray)
    //We still cannot get rid of this stupid singleton instance..
    //Todo: if we have a Renderer once, we can use this clean up opengl specific stuff
    SLGLProgramManager::deletePrograms();
    SLMaterialDefaultGray::deleteInstance();

    if (_closeCB)
    {
        LOG_ERLEBAR_DEBUG("Close Callback!");
        _closeCB();
    }

    addEvent(new DoneEvent("ErlebARApp::DESTROY"));
}

void ErlebARApp::SELECTION(const sm::NoEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateExit)
        return;

    _selectionView->update();
}

void ErlebARApp::START_TEST(const sm::NoEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateExit)
        return;

    if (_camera->permissionGranted())
    {
        _testView->start();
        addEvent(new DoneEvent("ErlebARApp::START_TEST"));
    }

    assert(_startUpView != nullptr);
    _startUpView->update();
}

void ErlebARApp::TEST(const sm::NoEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateExit)
    {
        return;
    }

    if (stateEntry)
    {
        _testView->show();
    }
    _testView->update();
}

void ErlebARApp::TEST_RUNNER(const sm::NoEventData* data, const bool stateEntry, const bool stateExit)
{
    _testRunnerView->update();
}

void ErlebARApp::LOCATION_MAP(const ErlebarEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateExit)
        return;

    if (stateEntry)
    {
        if (data)
            _locationMapView->initLocation(data->location);
    }

    _locationMapView->update();
}

void ErlebARApp::AREA_INFO(const AreaEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateExit)
        return;

    if (stateEntry)
    {
        _areaInfoView->show();
        //We use the area info view to initialize the area tracking. It is a convention
        //for this state, that if there is no data sent, we assume that the previous state was HOLD_TRACKING.
        if (data)
        {
            _areaInfoView->initArea(data->locId, data->areaId);
            _areaTrackingView->initArea(data->locId, data->areaId);
        }
        else
        {
            _areaTrackingView->resume();
        }
    }

    _areaInfoView->update();
}

void ErlebARApp::AREA_TRACKING(const AreaEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateExit)
        return;

    _areaTrackingView->update();
}

void ErlebARApp::HOLD_TRACKING(const sm::NoEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateEntry)
        _areaTrackingView->hold();
    else if (stateExit)
        _areaTrackingView->resume();
}

void ErlebARApp::TUTORIAL(const sm::NoEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateExit)
        return;

    _tutorialView->update();
}

void ErlebARApp::ABOUT(const sm::NoEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateExit)
        return;

    if (stateEntry)
    {
        _aboutView->show();
    }

    _aboutView->update();
}

void ErlebARApp::SETTINGS(const sm::NoEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateExit)
        return;

    if (stateEntry)
    {
        _settingsView->show();
    }

    _settingsView->update();
}

void ErlebARApp::CAMERA_TEST(const sm::NoEventData* data, const bool stateEntry, const bool stateExit)
{
    if (stateExit)
    {
        _cameraTestView->stopCamera();
        return;
    }

    if (stateEntry)
    {
        if (!_cameraTestView)
        {
            _cameraTestView = new CameraTestView(*this,
                                                 _inputManager,
                                                 *_imGuiEngine,
                                                 *_resources,
                                                 _camera,
                                                 *_dd);
        }

        _cameraTestView->show();
    }

    _cameraTestView->update();
}
