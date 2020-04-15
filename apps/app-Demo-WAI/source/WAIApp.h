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

#ifndef WAI_APP_H
#define WAI_APP_H

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
#include <SLTransformationNode.h>
#include <SLInputManager.h>
#include <WAIEvent.h>
#include <SlamParams.h>

//#include <sm/Event.h>
//#include <sm/EventHandler.h>
#include <sm/EventSender.h>
#include <sm/StateMachine.h>

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

//-----------------------------------------------------------------------------
// State machine impl
//-----------------------------------------------------------------------------
class ABCView;
class XYView;
class ABCEventData;

class WAIApp : public SLInputEventInterface
  , public sm::StateMachine
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
      : SLInputEventInterface(_inputManager),
        sm::StateMachine((unsigned int)StateId::IDLE)
    {
        registerState<WAIApp, sm::NoEventData, &WAIApp::stateIdle>((unsigned int)StateId::IDLE);
        registerState<WAIApp, sm::NoEventData, &WAIApp::stateInit>((unsigned int)StateId::INIT);
        registerState<WAIApp, ABCEventData, &WAIApp::stateProcessXY>((unsigned int)StateId::PROCESS_XY);
        registerState<WAIApp, sm::NoEventData, &WAIApp::stateProcessABC>((unsigned int)StateId::PROCESS_ABC);
        registerState<WAIApp, sm::NoEventData, &WAIApp::stateStop>((unsigned int)StateId::STOP);
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
    void stateIdle(const sm::NoEventData* data);
    void stateInit(const sm::NoEventData* data);
    void stateProcessXY(const ABCEventData* data);
    void stateProcessABC(const sm::NoEventData* data);
    void stateStop(const sm::NoEventData* data);

    std::string    _name;
    SLInputManager _inputManager;

    ABCView* _abcView = nullptr;
    XYView*  _xyView  = nullptr;
};

//-----------------------------------------------------------------------------
// Eventdata
//-----------------------------------------------------------------------------
class ABCEventData : public sm::EventData
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
class GoBackEvent : public sm::Event
{
public:
    //definition of possible transitions
    GoBackEvent()
    {
        enableTransition((unsigned int)WAIApp::StateId::PROCESS_XY, (unsigned int)WAIApp::StateId::PROCESS_ABC);
        enableTransition((unsigned int)WAIApp::StateId::PROCESS_XY, (unsigned int)WAIApp::StateId::PROCESS_ABC);
        enableTransition((unsigned int)WAIApp::StateId::PROCESS_ABC, (unsigned int)WAIApp::StateId::STOP);
    }
};

//go from idle to init state
class InitEvent : public sm::Event
{
public:
    //definition of possible transitions
    InitEvent()
    {
        enableTransition((unsigned int)WAIApp::StateId::IDLE, (unsigned int)WAIApp::StateId::INIT);
    }
};

//a state wants to be finished
class StateDoneEvent : public sm::Event
{
public:
    //definition of possible transitions
    StateDoneEvent()
    {
        enableTransition((unsigned int)WAIApp::StateId::INIT, (unsigned int)WAIApp::StateId::PROCESS_ABC);
        enableTransition((unsigned int)WAIApp::StateId::STOP, (unsigned int)WAIApp::StateId::IDLE);
    }
};

//a state wants to be finished
class StateABCDoneEvent : public sm::Event
{
public:
    //definition of possible transitions
    StateABCDoneEvent(std::string msg)
    {
        enableTransition((unsigned int)WAIApp::StateId::PROCESS_ABC, (unsigned int)WAIApp::StateId::PROCESS_XY);

        _eventData = new ABCEventData(msg);
    }
};

//-----------------------------------------------------------------------------
// Views
//-----------------------------------------------------------------------------

//ein state der einen event sendet
class XYView : public View
  , public sm::EventSender
{
public:
    XYView(sm::EventHandler& handler)
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
  , public sm::EventSender
{
public:
    ABCView(sm::EventHandler& handler)
      : sm::EventSender(handler)
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
//        started switch to START_UP state or directly start StartUpScene if Selection is already not Selection::NONE.
//        */
//        IDLE,
//        /*!In this state the user has to select a an Selection. When selection is not NONE, we switch to state START_UP
//        */
//        SELECTION,
//        /*!We start up the states depending on selected Selection
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
//    //done after Selection was selected, then we know what to start up
//    bool _startFromStartUp = false;
//
//    //defines if ErlebAR scene was already selected or if user has to choose
//    Selection _appMode            = Selection::NONE;
//    Area    _area               = Area::NONE;
//    bool    _showSelectionState = false;
//
//    bool _switchToTracking = false;
//
//    CloseAppCallback _closeCB = nullptr;
//    bool             _goBack  = false;
//};

#endif
