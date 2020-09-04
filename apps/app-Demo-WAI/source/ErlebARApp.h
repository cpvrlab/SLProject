#ifndef ERLEBAR_APP_H
#define ERLEBAR_APP_H

#include <mutex>
#include <sm/StateMachine.h>
#include <SLInputEventInterface.h>
#include <SLInputManager.h>
#include <ErlebAR.h>
#include <sens/SENSCamera.h>
#include <sens/SENSGps.h>
#include <Resources.h>
#include <ErlebAREvents.h>

class InitEventData;
class SelectionView;
class TestView;
class TestRunnerView;
class StartUpView;
class WelcomeView;
class SettingsView;
class AboutView;
class TutorialView;
class LocationMapView;
class AreaInfoView;
class AreaTrackingView;
class CameraTestView;
class SensorTestView;
class ImGuiEngine;

class ErlebARApp : public sm::StateMachine
  , public SLInputEventInterface
{
    using CloseAppCallback = std::function<void(void)>;

public:
    ErlebARApp();

    //external events:
    void init(int                scrWidth,
              int                scrHeight,
              int                dpi,
              const std::string& dataDir,
              const std::string& writableDir,
              SENSCamera*        camera,
              SENSGps*           gps);
    //go back (e.g. from android back-button)
    void goBack();
    //call to completely uninitialize app
    void destroy();
    //call when app goes into background but is not destroyed
    void hold();
    //call when app comes to foreground after being hold
    void resume();

    //! set a callback function which can be used to inform caller that app wants to be closed
    void setCloseAppCallback(CloseAppCallback cb) { _closeCB = cb; }

private:
    std::string getPrintableState(unsigned int state) override;

    void IDLE(const sm::NoEventData* data, const bool stateEntry, const bool stateExit);
    void INIT(const InitEventData* data, const bool stateEntry, const bool stateExit);
    void WELCOME(const sm::NoEventData* data, const bool stateEntry, const bool stateExit);
    void DESTROY(const sm::NoEventData* data, const bool stateEntry, const bool stateExit);
    void SELECTION(const sm::NoEventData* data, const bool stateEntry, const bool stateExit);

    void START_TEST(const sm::NoEventData* data, const bool stateEntry, const bool stateExit);
    void TEST(const sm::NoEventData* data, const bool stateEntry, const bool stateExit);
    void TEST_RUNNER(const sm::NoEventData* data, const bool stateEntry, const bool stateExit);

    void LOCATION_MAP(const ErlebarEventData* data, const bool stateEntry, const bool stateExit);
    void AREA_INFO(const AreaEventData* data, const bool stateEntry, const bool stateExit);
    void AREA_TRACKING(const AreaEventData* data, const bool stateEntry, const bool stateExit);
    void HOLD_TRACKING(const sm::NoEventData* data, const bool stateEntry, const bool stateExit);

    void TUTORIAL(const sm::NoEventData* data, const bool stateEntry, const bool stateExit);
    void ABOUT(const sm::NoEventData* data, const bool stateEntry, const bool stateExit);
    void SETTINGS(const sm::NoEventData* data, const bool stateEntry, const bool stateExit);
    void CAMERA_TEST(const sm::NoEventData* data, const bool stateEntry, const bool stateExit);
    void SENSOR_TEST(const sm::NoEventData* data, const bool stateEntry, const bool stateExit);

    SLInputManager              _inputManager;
    std::unique_ptr<DeviceData> _dd;
    SENSCamera*                 _camera           = nullptr;
    SENSGps*                    _gps              = nullptr;
    SelectionView*              _selectionView    = nullptr;
    TestView*                   _testView         = nullptr;
    TestRunnerView*             _testRunnerView   = nullptr;
    StartUpView*                _startUpView      = nullptr;
    WelcomeView*                _welcomeView      = nullptr;
    AboutView*                  _aboutView        = nullptr;
    SettingsView*               _settingsView     = nullptr;
    TutorialView*               _tutorialView     = nullptr;
    LocationMapView*            _locationMapView  = nullptr;
    AreaInfoView*               _areaInfoView     = nullptr;
    AreaTrackingView*           _areaTrackingView = nullptr;
    CameraTestView*             _cameraTestView   = nullptr;
    SensorTestView*             _sensorTestView   = nullptr;

    CloseAppCallback _closeCB = nullptr;

    ErlebAR::Resources* _resources   = nullptr;
    ImGuiEngine*        _imGuiEngine = nullptr;
};

#endif
