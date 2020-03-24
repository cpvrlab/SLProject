#ifndef ERLEBAR_APP_H
#define ERLEBAR_APP_H

#include <sm/StateMachine.h>
#include <SLInputEventInterface.h>
#include <SLInputManager.h>
#include <DeviceData.h>

class InitData;

class ErlebARApp : public sm::StateMachine
  , public SLInputEventInterface
{
public:
    enum class StateId
    {
        IDLE = 0,
        INIT,
        TERMINATE,
        SELECTION,
        MAP_VIEW,
        AREA_TRACKING,
        TEST
    };

    ErlebARApp()
      : sm::StateMachine((unsigned int)StateId::IDLE),
        SLInputEventInterface(_inputManager)
    {
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::IDLE>((unsigned int)StateId::IDLE);
        registerState<ErlebARApp, InitData, &ErlebARApp::INIT>((unsigned int)StateId::INIT);
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::TERMINATE>((unsigned int)StateId::TERMINATE);
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::SELECTION>((unsigned int)StateId::SELECTION);
        registerState<ErlebARApp, sm::NoEventData, &ErlebARApp::TEST>((unsigned int)StateId::TEST);
    }

    //external events:
    void init(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi, AppDirectories dirs);
    void goBack();

private:
    void IDLE(const sm::NoEventData* data);
    void INIT(const InitData* data);
    void TERMINATE(const sm::NoEventData* data);
    void SELECTION(const sm::NoEventData* data);
    void TEST(const sm::NoEventData* data);

    SLInputManager _inputManager;
};

//-----------------------------------------------------------------------------
// EventData
//-----------------------------------------------------------------------------
class InitData : public sm::EventData
{
public:
    InitData(DeviceData deviceData)
      : deviceData(deviceData)
    {
    }
    const DeviceData deviceData;
};

//-----------------------------------------------------------------------------
// Event
//-----------------------------------------------------------------------------
class InitEvent : public sm::Event
{
public:
    InitEvent(int            scrWidth,
              int            scrHeight,
              float          scr2fbX,
              float          scr2fbY,
              int            dpi,
              AppDirectories dirs)
    {
        enableTransition((unsigned int)ErlebARApp::StateId::IDLE,
                         (unsigned int)ErlebARApp::StateId::INIT);

        DeviceData deviceData(scrWidth, scrHeight, scr2fbX, scr2fbY, dpi, dirs);
        _eventData = new InitData(deviceData);
    }
};

class GoBackEvent : public sm::Event
{
public:
    GoBackEvent()
    {
        enableTransition((unsigned int)ErlebARApp::StateId::SELECTION,
                         (unsigned int)ErlebARApp::StateId::TERMINATE);
        enableTransition((unsigned int)ErlebARApp::StateId::AREA_TRACKING,
                         (unsigned int)ErlebARApp::StateId::MAP_VIEW);
        enableTransition((unsigned int)ErlebARApp::StateId::MAP_VIEW,
                         (unsigned int)ErlebARApp::StateId::SELECTION);
    }
};

class StateDoneEvent : public sm::Event
{
public:
    StateDoneEvent()
    {
        enableTransition((unsigned int)ErlebARApp::StateId::TERMINATE,
                         (unsigned int)ErlebARApp::StateId::IDLE);
    }
};

#endif
