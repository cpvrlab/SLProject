#ifndef ERLEBAR_APP_H
#define ERLEBAR_APP_H

#include <mutex>
#include <sm/StateMachine.h>
#include <SLInputEventInterface.h>
#include <SLInputManager.h>
#include <ErlebAR.h>
#include <sens/SENSCamera.h>

class InitData;
class SelectionView;
class TestView;
class StartUpView;
class WelcomeView;
class View;

class ErlebARApp : public sm::StateMachine
  , public SLInputEventInterface
{
    using CloseAppCallback = std::function<void(void)>;

public:
    ErlebARApp()
      : sm::StateMachine((unsigned int)StateId::IDLE),
        SLInputEventInterface(_inputManager)
    {
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::IDLE>((unsigned int)StateId::IDLE);
        registerState<ErlebARApp, InitData, &ErlebARApp::INIT>((unsigned int)StateId::INIT);
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::WELCOME>((unsigned int)StateId::WELCOME);
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::DESTROY>((unsigned int)StateId::DESTROY);
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::SELECTION>((unsigned int)StateId::SELECTION);

        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::START_TEST>((unsigned int)StateId::START_TEST);
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::TEST>((unsigned int)StateId::TEST);
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::HOLD_TEST>((unsigned int)StateId::HOLD_TEST);
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::RESUME_TEST>((unsigned int)StateId::RESUME_TEST);

        registerState<ErlebARApp, ErlebarData, &ErlebARApp::START_ERLEBAR>((unsigned int)StateId::START_ERLEBAR);
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::MAP_VIEW>((unsigned int)StateId::MAP_VIEW);
        registerState<ErlebARApp, AreaData, &ErlebARApp::AREA_TRACKING>((unsigned int)StateId::AREA_TRACKING);

        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::TUTORIAL>((unsigned int)StateId::TUTORIAL);
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::ABOUT>((unsigned int)StateId::ABOUT);
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::CAMERA_TEST>((unsigned int)StateId::CAMERA_TEST);
    }

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
    void CAMERA_TEST(const sm::NoEventData* data, const bool stateEntry);

    SLInputManager _inputManager;

    SelectionView* _selectionView = nullptr;
    TestView*      _testView      = nullptr;
    StartUpView*   _startUpView   = nullptr;
    WelcomeView*   _welcomeView   = nullptr;
    View*          _currentView   = nullptr;

    SENSCamera*      _camera  = nullptr;
    CloseAppCallback _closeCB = nullptr;
};

#endif
