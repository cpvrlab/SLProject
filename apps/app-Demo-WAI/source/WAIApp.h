//#############################################################################
//  File:      WAISceneView.h
//  Purpose:   Node transform test application that demonstrates all transform
//             possibilities of SLNode
//  Author:    Marc Wacker
//  Date:      July 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef APP_WAI_SCENE_VIEW
#define APP_WAI_SCENE_VIEW

#include <vector>
#include <functional>
#include "AppWAIScene.h"

#include <CVCalibration.h>
#include <WAIAutoCalibration.h>
#include <DeviceData.h>
#include <AppDemoWaiGui.h>
#include <SLInputEventInterface.h>
#include <WAISlam.h>
#include <SENSCamera.h>
#include <SENSVideoStream.h>
#include <GLSLextractor.h>
#include <FeatureExtractorFactory.h>
#include <SLInputManager.h>
#include <WAIEvent.h>
#include <SlamParams.h>

//#include <states/SelectionState.h>
//#include <states/StartUpState.h>
//#include <states/AreaTrackingState.h>
//#include <states/CameraTestState.h>
//#include <states/LocationMapState.h>
//#include <states/TestState.h>
//#include <states/TutorialState.h>

class SLMaterial;
class SLPoints;
class SLNode;
class AppDemoGuiError;
class AppDemoGuiSlamLoad;

class View
{
public:
    ~View()
    {
        _startThread.detach();
    }
    //! asynchronous start
    void start()
    {
        if (_started)
            return;

        _startThread = std::thread(&View::doStart, this);
    }

    //! if ready the state machine can change to this state
    bool started() { return _started; }
    //! signalizes that state is ready and wants caller to switch to another state
    bool ready() { return _ready; }
    void setStateReady() { _ready = true; }

    //! update this state
    virtual bool update() = 0;

protected:
    //! implement startup functionality here. Set _started to true when done.
    virtual void doStart(){};

    //set to true if startup is done
    bool _started = false;

    //! signalizes that state is ready and wants caller to switch to another state
    bool _ready = false;

private:
    std::thread _startThread;
};

namespace SM
{

class EventData
{
public:
    virtual ~EventData() {}
};

//event base class
class Event
{
public:
    //default states
    enum
    {
        EVENT_IGNORED = 0xFE,
        CANNOT_HAPPEN
    };

    virtual ~Event(){};

    unsigned int getNewState(unsigned int currentState)
    {
        auto it = _transitions.find(currentState);
        if (it != _transitions.end())
        {
            return it->second;
        }
        else
        {
            return EVENT_IGNORED;
        }
    }

protected:
    std::map<unsigned int, unsigned int> _transitions;
};

class EventHandler
{
public:
    void addEvent(Event* e)
    {
        _events.push(e);
    }

protected:
    std::queue<Event*> _events;
};

//state is event sender
class EventSender
{
public:
    EventSender(EventHandler& handler)
      : _handler(handler)
    {
    }
    EventSender() = delete;
    void sendEvent(Event* event)
    {
        _handler.addEvent(event);
    }

private:
    EventHandler& _handler;
};

class StateMachine : public EventHandler
{
public:
    virtual ~StateMachine(){};

    unsigned int currentState() { return _currentStateId; }

    void handleEvents()
    {
        while (_events.size())
        {
            Event* e = _events.front();
            _events.pop();

            unsigned int newState = e->getNewState(_currentStateId);
            if (newState != Event::EVENT_IGNORED)
            {
                _currentStateId = newState;
            }
            else
            {
                std::cout << "Event ignored" << std::endl;
                //delete event data
            }

            delete e;
        }
    }

protected:
    unsigned int _currentStateId = 0;
};

} //namespace SM

//-----------------------------------------------------------------------------
// Views
//-----------------------------------------------------------------------------

//ein state der einen event sendet
class XYView : public View
  , public SM::EventSender
{
public:
    XYView(SM::EventHandler& handler)
      : EventSender(handler)
    {
    }

    bool update() override
    {
        return false;
    }
};

//ein state der einen event sendet
class ABCView : public View
  , public SM::EventSender
{
public:
    ABCView(SM::EventHandler& handler)
      : SM::EventSender(handler)
    {
    }

    bool update() override
    {
        return false;
    }
};

//-----------------------------------------------------------------------------
// State machine impl
//-----------------------------------------------------------------------------
class InitEvent;

class WAIApp : public SLInputEventInterface
  , public SM::StateMachine
{
    friend InitEvent;

public:
    enum class StateId
    {
        IDLE = 0,
        INIT,
        PROCESS_XY,
        PROCESS_ABC,
        STOP
    };

    //static std::map<StateId, std::function<void(void)>> state =
    //{
    //  {StateId::IDLE, std::bind(&WAIApp::stateIdle, this)},
    //  {StateId::INIT, std::bind(&WAIApp::stateInit, this)},
    //  {StateId::PROCESS_XY, std::bind(&WAIApp::stateProcessXY, this)},
    //  {StateId::PROCESS_ABC, std::bind(&WAIApp::stateProcessABC, this)}};

    WAIApp()
      : SLInputEventInterface(_inputManager)
    {
        _currentStateId = (unsigned int)StateId::IDLE;
    }

    bool update()
    {
        handleEvents();
        StateId state = (StateId)_currentStateId;
        switch (state)
        {
            case StateId::IDLE:
                stateIdle();
                break;
            case StateId::INIT:
                stateInit();
                break;
            case StateId::PROCESS_ABC:
                stateProcessABC();
                break;
            case StateId::PROCESS_XY:
                stateProcessXY();
                break;
            case StateId::STOP:
                stateStop();
                break;
        }

        return true;
    }

    //external events:

    void load(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi, AppDirectories directories);
    void goBack();

    std::string name()
    {
        return _name;
    }

private:
    //state update functions corresponding to the states defined above
    void stateIdle();
    void stateInit(/*InitEventData* data*/);
    void stateProcessXY();
    void stateProcessABC();
    void stateStop();

    std::string    _name;
    SLInputManager _inputManager;

    ABCView* _abcView = nullptr;
    XYView*  _xyView  = nullptr;
};

//-----------------------------------------------------------------------------
// Events
//-----------------------------------------------------------------------------

//go back from TestState leads to update call of
class GoBackEvent : public SM::Event
{
public:
    //definition of possible transitions
    GoBackEvent()
    {
        _transitions[(unsigned int)WAIApp::StateId::PROCESS_XY]  = (unsigned int)WAIApp::StateId::PROCESS_ABC;
        _transitions[(unsigned int)WAIApp::StateId::PROCESS_ABC] = (unsigned int)WAIApp::StateId::STOP;
    }
};

//go from idle to init state
class InitEvent : public SM::Event
{
public:
    //definition of possible transitions
    InitEvent()
    {
        _transitions[(unsigned int)WAIApp::StateId::IDLE] = (unsigned int)WAIApp::StateId::INIT;
    }
};

//a state wants to be finished
class StateDoneEvent : public SM::Event
{
public:
    //definition of possible transitions
    StateDoneEvent()
    {
        _transitions[(unsigned int)WAIApp::StateId::INIT]        = (unsigned int)WAIApp::StateId::PROCESS_ABC;
        _transitions[(unsigned int)WAIApp::StateId::PROCESS_ABC] = (unsigned int)WAIApp::StateId::PROCESS_XY;
        _transitions[(unsigned int)WAIApp::StateId::STOP]        = (unsigned int)WAIApp::StateId::IDLE;
    }
};

//class WAIApp : public SLInputEventInterface
//  , public StateMachine
//{
//public:
//    using CloseAppCallback = std::function<void(void)>;
//    enum class State
//    {
//        /*!Wait for _startFromIdle to become true. When it is true start SelectionState for scene selection and when it is
//        started switch to START_UP state or directly start StartUpScene if AppMode is already not AppMode::NONE.
//        */
//        IDLE,
//        /*!In this state the user has to select a an AppMode. When selection is not NONE, we switch to state START_UP
//        */
//        SELECTION,
//        /*!We start up the states depending on selected AppMode
//        */
//        START_UP,
//        LOCATION_MAP,
//        AREA_TRACKING,
//        TEST,
//        CAMERA_TEST,
//        TUTORIAL,
//        TERMINATE
//    };
//
//    WAIApp();
//    ~WAIApp();
//    //call load to correctly initialize wai app
//    int  load(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi, AppDirectories directories);
//    void setCamera(SENSCamera* camera);
//    //call update to update the frame, wai and visualization
//    bool update();
//    void close();
//    void terminate();
//
//    std::string name();
//
//    //! set a callback function which can be used to inform caller that app wants to be closed
//    void setCloseAppCallback(CloseAppCallback cb) { _closeCB = cb; }
//    //! caller informs that app back button was pressed
//    void goBack()
//    {
//        _goBack = true;
//    }
//
//private:
//    SENSCamera* getCamera();
//
//    void reset();
//    void checkStateTransition();
//    bool updateState();
//
//    AppDirectories _dirs;
//
//    SENSCamera* _camera = nullptr;
//    std::mutex  _cameraSetMutex;
//
//    std::string    _name;
//    SLInputManager _inputManager;
//
//    std::unique_ptr<DeviceData> _deviceData;
//
//    State              _state             = State::IDLE;
//    SelectionState*    _selectionState    = nullptr;
//    StartUpState*      _startUpState      = nullptr;
//    AreaTrackingState* _areaTrackingState = nullptr;
//    CameraTestState*   _cameraTestState   = nullptr;
//    LocationMapState*  _locationMapState  = nullptr;
//    TestState*         _testState         = nullptr;
//    TutorialState*     _tutorialState     = nullptr;
//
//    //Sub-States that lead to state start() call:
//    //set to true as soon as we have access to app resouces, then we can first visualize something
//    bool _startFromIdle = false;
//    //done after AppMode was selected, then we know what to start up
//    bool _startFromStartUp = false;
//
//    //defines if ErlebAR scene was already selected or if user has to choose
//    AppMode _appMode            = AppMode::NONE;
//    Area    _area               = Area::NONE;
//    bool    _showSelectionState = false;
//
//    bool _switchToTracking = false;
//
//    CloseAppCallback _closeCB = nullptr;
//    bool             _goBack  = false;
//};

#endif
