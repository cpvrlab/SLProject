#ifndef ERLEBAR_USERGUIDANCE_H
#define ERLEBAR_USERGUIDANCE_H

#include <scenes/UserGuidanceScene.h>
#include <ErlebAR.h>
#include <AreaTrackingGui.h>
#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>
#include <Resources.h>

//user guidance for AreaTrackingView
class UserGuidance
{
    enum class State
    {
        IDLE=0,
        DATA_LOADING,
        DIR_ARROW,
        RELOCALIZING, //this is the state for user guidance, maybe view always tries to relocalize
        RELOCALIZING_WRONG_ORIENTATION,
        TRACKING
    };
    
    const char* mapStateToString(State state)
    {
        switch (state)
        {
            case State::IDLE:
                return "NONE";
            case State::DATA_LOADING:
                return "DATA_LOADING";
            case State::DIR_ARROW:
                return "DIR_ARROW";
            case State::RELOCALIZING:
                return "RELOCALIZING";
            case State::RELOCALIZING_WRONG_ORIENTATION:
                return "RELOCALIZING_WRONG_ORIENTATION";
            case State::TRACKING:
                return "TRACKING";
            default:
                return "Missing state to name mapping!";
        }
    }
    
public:
    UserGuidance(UserGuidanceScene*  userGuidanceScene,
                 AreaTrackingGui*    gui,
                 SENSGps*            gps,
                 SENSOrientation*    orientation,
                 ErlebAR::Resources& resources);

    //reset user guidance to state idle
    void reset();
    //update userguidance
    void areaSelected(ErlebAR::AreaId areaId, SLVec3d areaLocation, float areaOrientation);
    void dataIsLoading(bool isLoading);
    void updateTrackingState(bool isTracking);
    //update values for state updates and transitions depending on sensors
    void updateSensorEstimations();
    
private:
    //update the finite state maching
    void update();
    
    //check if a state transition is met and change to new state.
    //The necessary state transition depends on the current state.
    //If a transition is met certain things may be changed depending on the last state.
    bool stateTransition();
    //process state exit from old state
    void processStateExit();
    //process state entry into new state
    void processStateEntry();
    //update the current state
    void processState();
    
    void estimateEcefToEnuRotation();
    void estimateArrowOrientation();
    
    SENSGps* _gps = nullptr;
    SENSOrientation* _orientation = nullptr;
    //interface to update orientation of direction arrow
    UserGuidanceScene* _userGuidanceScene = nullptr;
    AreaTrackingGui* _gui = nullptr;
    const ErlebAR::Resources& _resources;
    
    //HELPER VARIABLES:
    // distance to area threshold in meter
    const float DIST_TO_AREA_THRES_M = 10.f;
    SLVec3d _areaLocation{0.f, 0.f, 0.f};
    float _areaOrientation = 0.f;
    float _currentDistanceToAreaM = 0.f;
    SLVec3d _ecefAREA;
    SLMat3d _enuRecef;
    SLVec3d _ecefDEVICE;
    // orientation difference to area threshold
    const float ORIENT_DIFF_TO_AREA_THRES_DEG = 10.f;
    float _currentOrientationDeg = 0.f;
    bool _isTracking = false;
    
    //STATE VARIABLES: state transitions depend on these values. They all have to be resetted in reset() function
    ErlebAR::AreaId _selectedArea = ErlebAR::AreaId::NONE;
    bool _dataIsLoading = false;
    bool _highDistanceToArea = false;
    bool _wrongOrientation = false;
    //current state
    State _state = State::IDLE;
    State _oldState = State::IDLE;
};

#endif

/*
#include <functional>
#include <WAISlamTools.h>
#include <HighResTimer.h>

class AreaTrackingGui;

struct UserGuidanceInfo
{
    UserGuidanceInfo()
    {
        _started = false;
    }
    void terminate() { _terminate = true; }

    bool update(float timeNow, AreaTrackingGui* gui)
    {
        if (!_started)
        {
            _timeStart = timeNow;
            _started   = true;
            _terminate = false;
        }

        if (!_terminate)
            _timeTerminate = timeNow;

        if (!updateFct(timeNow - _timeStart, timeNow - _timeTerminate, gui, _terminate))
        {
            _started = false;
            return false;
        }
        return true;
    }

    bool  _started;
    bool  _terminate;
    float _timeStart;
    float _timeTerminate;

    std::function<bool(float, float, AreaTrackingGui*, bool&)> updateFct;
};

class UserGuidance
{
public:
    UserGuidance(AreaTrackingGui* gui);
    void update(WAI::TrackingState state);

private:
    void flush();

    AreaTrackingGui*   _gui;
    WAI::TrackingState _lastWAIState;
    HighResTimer       _timer;

    UserGuidanceInfo _alignImgInfo;
    UserGuidanceInfo _moveLeftRight;
    UserGuidanceInfo _trackingMarker;
    UserGuidanceInfo _trackingStarted;

    std::queue<UserGuidanceInfo*> _queuedInfos;

    bool _marker = false;
    bool _trackedOnce = false;
};
*/


