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

class NoEventData : public EventData
{
public:
    NoEventData()
    {
    }
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

    EventData* getEventData()
    {
        return _eventData;
    }

protected:
    std::map<unsigned int, unsigned int> _transitions;
    EventData*                           _eventData = nullptr;
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

class StateMachine;

/// @brief Abstract state base class that all states inherit from.
class StateBase
{
public:
    /// Called by the state machine engine to execute a state action. If a guard condition
    /// exists and it evaluates to false, the state action will not execute.
    /// @param[in] sm - A state machine instance.
    /// @param[in] data - The event data.
    virtual void InvokeStateAction(StateMachine* sm, const EventData* data) const {};
};

/* @brief StateAction takes three template arguments: A state machine class,
          a state function event data type (derived from EventData) and a state machine
*/
member function pointer.template<class SM, class Data, void (SM::*Func)(const Data*)>
class StateAction : public StateBase
{
public:
    virtual void InvokeStateAction(StateMachine* sm, const EventData* data) const
    {
        // Downcast the state machine and event data to the correct derived type
        SM* derivedSM = static_cast<SM*>(sm);

        const Data* derivedData = dynamic_cast<const Data*>(data);

        // Call the state function
        (derivedSM->*Func)(derivedData);
    }
};

class StateMachine : public EventHandler
{
public:
    virtual ~StateMachine()
    {
        for (auto it : _stateActions)
        {
            delete it.second;
        }
    };

    unsigned int currentState() { return _currentStateId; }

    template<class SM, class Data, void (SM::*Func)(const Data*)>
    void registerState(unsigned int stateId)
    {
        StateBase* sb          = new StateAction<SM, Data, Func>();
        _stateActions[stateId] = sb;
    }

    bool update()
    {
        SM::EventData* data = nullptr;
        //we only handle one event per update call!
        if (_events.size())
        {
            Event* e = _events.front();
            _events.pop();

            unsigned int newState = e->getNewState(_currentStateId);
            data                  = e->getEventData();
            if (newState != Event::EVENT_IGNORED)
            {
                _currentStateId = newState;
            }
            else
            {
                Utils::log("StateMachine", "Event ignored");
                //delete event data
            }

            delete e;
        }

        _stateActions[_currentStateId]->InvokeStateAction(this, data);
        //const SM::StateBase* stateMap = getStateMap();
        //stateMap[_currentStateId].InvokeStateAction(this, data);

        return true;
    }

protected:
    /// Gets the state map as defined in the derived class. The BEGIN_STATE_MAP,
    /// STATE_MAP_ENTRY and END_STATE_MAP macros are used to assist in creating the
    /// map. A state machine only needs to return a state map using either GetStateMap()
    /// or GetStateMapEx() but not both.
    /// @return An array of StateMapRow pointers with the array size MAX_STATES or
    /// NULL if the state machine uses the GetStateMapEx().
    //virtual const StateBase* getStateMap() = 0;

    unsigned int                           _currentStateId = 0;
    std::map<unsigned int, SM::StateBase*> _stateActions;
    //std::vector<SM::StateBase*> _stateActions;
};

} //namespace SM

//-----------------------------------------------------------------------------
// State machine impl
//-----------------------------------------------------------------------------
class InitEvent;
class ABCView;
class XYView;
class ABCEventData;

class WAIApp : public SLInputEventInterface
  , public SM::StateMachine
{
public:
    enum class StateId
    {
        IDLE = 0,
        INIT,
        PROCESS_XY,
        PROCESS_ABC,
        STOP,
        StateId_END
    };

    WAIApp()
      : SLInputEventInterface(_inputManager)
    {
        _currentStateId = (unsigned int)StateId::IDLE;

        registerState<WAIApp, SM::NoEventData, &WAIApp::stateIdle>((unsigned int)StateId::IDLE);
        registerState<WAIApp, SM::NoEventData, &WAIApp::stateInit>((unsigned int)StateId::INIT);
        registerState<WAIApp, ABCEventData, &WAIApp::stateProcessXY>((unsigned int)StateId::PROCESS_XY);
        registerState<WAIApp, SM::NoEventData, &WAIApp::stateProcessABC>((unsigned int)StateId::PROCESS_ABC);
        registerState<WAIApp, SM::NoEventData, &WAIApp::stateStop>((unsigned int)StateId::STOP);
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
    void stateIdle(const SM::NoEventData* data);
    void stateInit(const SM::NoEventData* data);
    void stateProcessXY(const ABCEventData* data);
    void stateProcessABC(const SM::NoEventData* data);
    void stateStop(const SM::NoEventData* data);

    std::string    _name;
    SLInputManager _inputManager;

    ABCView* _abcView = nullptr;
    XYView*  _xyView  = nullptr;
};

//-----------------------------------------------------------------------------
// Eventdata
//-----------------------------------------------------------------------------
class ABCEventData : public SM::EventData
{
public:
    ABCEventData(std::string msg)
      : msg(msg)
    {
    }

    std::string msg;
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
        _transitions[(unsigned int)WAIApp::StateId::INIT] = (unsigned int)WAIApp::StateId::PROCESS_ABC;
        _transitions[(unsigned int)WAIApp::StateId::STOP] = (unsigned int)WAIApp::StateId::IDLE;
    }
};

//a state wants to be finished
class StateABCDoneEvent : public SM::Event
{
public:
    //definition of possible transitions
    StateABCDoneEvent(std::string msg)
    {
        _transitions[(unsigned int)WAIApp::StateId::PROCESS_ABC] = (unsigned int)WAIApp::StateId::PROCESS_XY;

        _eventData = new ABCEventData(msg);
    }
};

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
        static int i = 0;
        i++;
        if (i == 5)
        {
            sendEvent(new StateABCDoneEvent("Do something!"));
        }
        return false;
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
