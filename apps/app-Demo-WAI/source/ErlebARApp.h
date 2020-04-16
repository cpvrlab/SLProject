#ifndef ERLEBAR_APP_H
#define ERLEBAR_APP_H

#include <mutex>
#include <sm/StateMachine.h>
#include <SLInputEventInterface.h>
#include <SLInputManager.h>
#include <ErlebAR.h>
#include <sens/SENSCamera.h>
#include <Resources.h>

class InitData;
class SelectionView;
class TestView;
class StartUpView;
class WelcomeView;
class SettingsView;
class AboutView;
class TutorialView;

class ErlebARApp : public sm::StateMachine
  , public SLInputEventInterface
{
    using CloseAppCallback = std::function<void(void)>;

public:
    ErlebARApp();

    //external events:
    void init(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi, AppDirectories dirs, SENSCamera* camera);
    //go back (e.g. from android back-button)
    void goBack();
    void destroy();
    void hold();
    void resume();

    //! set a callback function which can be used to inform caller that app wants to be closed
    void setCloseAppCallback(CloseAppCallback cb) { _closeCB = cb; }

private:
    void IDLE(const sm::NoEventData* data, const bool stateEntry);
    void INIT(const InitData* data, const bool stateEntry);
    void WELCOME(const sm::NoEventData* data, const bool stateEntry);
    void DESTROY(const sm::NoEventData* data, const bool stateEntry);
    void SELECTION(const sm::NoEventData* data, const bool stateEntry);

    void START_TEST(const sm::NoEventData* data, const bool stateEntry);
    void TEST(const sm::NoEventData* data, const bool stateEntry);
    void HOLD_TEST(const sm::NoEventData* data, const bool stateEntry);
    void RESUME_TEST(const sm::NoEventData* data, const bool stateEntry);

    void START_ERLEBAR(const ErlebarData* data, const bool stateEntry);
    void MAP_VIEW(const sm::NoEventData* data, const bool stateEntry);
    void AREA_TRACKING(const AreaData* data, const bool stateEntry);

    void TUTORIAL(const sm::NoEventData* data, const bool stateEntry);
    void ABOUT(const sm::NoEventData* data, const bool stateEntry);
    void SETTINGS(const sm::NoEventData* data, const bool stateEntry);
    void CAMERA_TEST(const sm::NoEventData* data, const bool stateEntry);

    SLInputManager _inputManager;

    SelectionView* _selectionView = nullptr;
    TestView*      _testView      = nullptr;
    StartUpView*   _startUpView   = nullptr;
    WelcomeView*   _welcomeView   = nullptr;
    AboutView*     _aboutView     = nullptr;
    SettingsView*  _settingsView  = nullptr;
    TutorialView*  _tutorialView  = nullptr;

    SENSCamera*      _camera  = nullptr;
    CloseAppCallback _closeCB = nullptr;

    ErlebAR::Resources* _resources = nullptr;
};

#endif
