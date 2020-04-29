#ifndef ERLEBAR_APP_H
#define ERLEBAR_APP_H

#include <mutex>
#include <sm/StateMachine.h>
#include <SLInputEventInterface.h>
#include <SLInputManager.h>
#include <ErlebAR.h>
#include <sens/SENSCamera.h>
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

class ErlebARApp : public sm::StateMachine
  , public SLInputEventInterface
{
    using CloseAppCallback = std::function<void(void)>;

public:
    ErlebARApp();

    //external events:
    void init(int scrWidth, int scrHeight, int dpi, AppDirectories dirs, SENSCamera* camera);
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

    void IDLE(const sm::NoEventData* data, const bool stateEntry);
    void INIT(const InitEventData* data, const bool stateEntry);
    void WELCOME(const sm::NoEventData* data, const bool stateEntry);
    void DESTROY(const sm::NoEventData* data, const bool stateEntry);
    void SELECTION(const sm::NoEventData* data, const bool stateEntry);

    void START_TEST(const sm::NoEventData* data, const bool stateEntry);
    void TEST(const sm::NoEventData* data, const bool stateEntry);
    void TEST_RUNNER(const sm::NoEventData* data, const bool stateEntry);
    void HOLD_TEST(const sm::NoEventData* data, const bool stateEntry);
    void RESUME_TEST(const sm::NoEventData* data, const bool stateEntry);

    void LOCATION_MAP(const ErlebarEventData* data, const bool stateEntry);
    void AREA_INFO(const AreaEventData* data, const bool stateEntry);
    void AREA_TRACKING(const AreaEventData* data, const bool stateEntry);
    void HOLD_TRACKING(const sm::NoEventData* data, const bool stateEntry);

    void TUTORIAL(const sm::NoEventData* data, const bool stateEntry);
    void ABOUT(const sm::NoEventData* data, const bool stateEntry);
    void SETTINGS(const sm::NoEventData* data, const bool stateEntry);
    void CAMERA_TEST(const sm::NoEventData* data, const bool stateEntry);

    SLInputManager _inputManager;

    SelectionView*    _selectionView    = nullptr;
    TestView*         _testView         = nullptr;
    TestRunnerView*   _testRunnerView   = nullptr;
    StartUpView*      _startUpView      = nullptr;
    WelcomeView*      _welcomeView      = nullptr;
    AboutView*        _aboutView        = nullptr;
    SettingsView*     _settingsView     = nullptr;
    TutorialView*     _tutorialView     = nullptr;
    LocationMapView*  _locationMapView  = nullptr;
    AreaInfoView*     _areaInfoView     = nullptr;
    AreaTrackingView* _areaTrackingView = nullptr;

    SENSCamera*      _camera  = nullptr;
    CloseAppCallback _closeCB = nullptr;

    ErlebAR::Resources* _resources = nullptr;
};

#endif
