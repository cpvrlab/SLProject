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
    
public:
    UserGuidance(UserGuidanceScene* userGuidanceScene, AreaTrackingGui* gui, SENSGps* gps, SENSOrientation* orientation, ErlebAR::Resources& resources)
     : _userGuidanceScene(userGuidanceScene),
       _gui(gui),
       _gps(gps),
       _orientation(orientation),
       _resources(resources)
    {
        reset();
    }

       
    //reset user guidance to state idle
    void reset()
    {
        _state = State::IDLE;
        _oldState = State::IDLE;
        _selectedArea = ErlebAR::AreaId::NONE;
        _dataIsLoading = false;
        _highDistanceToArea = false;
        _wrongOrientation = false;
        _isTracking = false;
        update();
    }
    
    //update userguidance
    void areaSelected(ErlebAR::AreaId areaId, SLVec3f areaLocation, float areaOrientation)
    {
        _selectedArea = areaId;
        _areaLocation = areaLocation;
        _areaOrientation = areaOrientation;
        update();
    }
    
    void dataIsLoading(bool isLoading)
    {
        if(isLoading != _dataIsLoading)
        {
            _dataIsLoading = isLoading;
            update();
        }
    }
    
    void updateTrackingState(bool isTracking)
    {
        if(isTracking != _isTracking)
        {
            _isTracking = isTracking;
            update();
        }
    }

    //update values for state updates and transitions depending on sensors
    void updateSensorEstimations()
    {
        //update distance to area and set _highDistanceToArea
        //_currentDistanceToAreaM = ...
        //todo: this change needs some filtering! We dont want state jittering at the border of thresholds
        float highDistanceToArea = (_currentDistanceToAreaM > DIST_TO_AREA_THRES_M);
        
        //update orientation
        //_currentYawOrientationDeg = ...
        //todo: this change needs some filtering! We dont want state jittering at the border of thresholds
        float wrongOrientation = (_currentOrientationDeg > ORIENT_DIFF_TO_AREA_THRES_DEG);
       
        //always call update to update arrow orientation
        update();
    }
    
private:
    void update()
    {
        //store old state
        _oldState = _state;
        //check for transitions to a new state and process the change
        while( stateTransition())
        {
        }
        
        if(_state != _oldState)
        {
            //process state exit from old state
            processStateExit();
            //process state entry into new state
            processStateEntry();
        }

        processState();
    }
    
    //check if a state transition is met and change to new state.
    //The necessary state transition depends on the current state.
    //If a transition is met certain things may be changed depending on the last state.
    bool stateTransition()
    {
        bool transition = false;
        
        switch (_state)
        {
            case State::IDLE: {
                if(_selectedArea != ErlebAR::AreaId::NONE)
                {
                    _state = State::DATA_LOADING;
                    transition = true;
                }
                break;
            }
            case State::DATA_LOADING: {
                if(_gps && _gps->permissionGranted() && _highDistanceToArea)
                {
                    _state = State::DIR_ARROW;
                    transition = true;
                }
                else if(!_dataIsLoading)
                {
                    _state = State::RELOCALIZING;
                    transition = true;
                }

                break;
            }
            case State::DIR_ARROW: {
                if(!_highDistanceToArea)
                {
                    if(_dataIsLoading)
                    {
                        _state = State::DATA_LOADING;
                        transition = true;
                    }
                    else
                    {
                        _state = State::RELOCALIZING;
                        transition = true;
                    }
                }
                break;
            }
            case State::RELOCALIZING: {
                if(_isTracking)
                {
                    _state = State::TRACKING;
                    transition = true;
                }
                else if(_gps && _gps->permissionGranted() && _highDistanceToArea)
                {
                    _state = State::DIR_ARROW;
                    transition = true;
                }
                else if(_dataIsLoading)
                {
                    _state = State::DATA_LOADING;
                    transition = true;
                }
                else if(_orientation && !_wrongOrientation)
                {
                    _state = State::RELOCALIZING_WRONG_ORIENTATION;
                    transition = true;
                }
                break;
            }
            case State::RELOCALIZING_WRONG_ORIENTATION: {
                if(_isTracking)
                {
                    _state = State::TRACKING;
                    transition = true;
                }
                else if(_dataIsLoading)
                {
                    _state = State::DATA_LOADING;
                    transition = true;
                }
                else if(_orientation && !_wrongOrientation)
                {
                    _state = State::RELOCALIZING;
                    transition = true;
                }
                break;
            }
            case State::TRACKING: {
                if(!_isTracking)
                {
                    _state = State::RELOCALIZING;
                    transition = true;
                }
                break;
            }
        }
        
        return transition;
    }
    
    //process state exit from old state
    void processStateExit()
    {
        //we check the old state, that we are leaving
        switch (_oldState)
        {
            case State::IDLE: {
                break;
            }
            case State::DATA_LOADING: {
                break;
            }
            case State::DIR_ARROW: {
                break;
            }
            case State::RELOCALIZING: {
                break;
            }
            case State::RELOCALIZING_WRONG_ORIENTATION: {
                break;
            }
            case State::TRACKING: {
                break;
            }
        }
    }
    //process state entry into new state
    void processStateEntry()
    {
        switch (_state)
        {
            case State::IDLE: {
                break;
            }
            case State::DATA_LOADING: {
                break;
            }
            case State::DIR_ARROW: {
                break;
            }
            case State::RELOCALIZING: {
                break;
            }
            case State::RELOCALIZING_WRONG_ORIENTATION: {
                break;
            }
            case State::TRACKING: {
                break;
            }
        }
    }
    
    //update the current state
    void processState()
    {
        switch (_state)
        {
            case State::IDLE: {
                //do nothing
                break;
            }
            case State::DATA_LOADING: {
                //show data loading indicator
                break;
            }
            case State::DIR_ARROW: {
                //update orientation calculation of direction arrow and update scene
                //update info bar with information about distance to target
                break;
            }
            case State::RELOCALIZING: {
                //help user to correctly move
                break;
            }
            case State::RELOCALIZING_WRONG_ORIENTATION: {
                break;
            }
            case State::TRACKING: {
                break;
            }
        }
    }
    
    void estimateArrowOrientation()
    {
        //todo: direction depending on current position and area position
        
        ///////////////////////////////////////////////////////////////////////
        // Build pose of camera in world frame (scene) using device rotation //
        ///////////////////////////////////////////////////////////////////////

        //camera rotation with respect to (w.r.t.) sensor
        SLMat3f sRc;
        sRc.rotation(-90, 0, 0, 1);

        //sensor rotation w.r.t. east-north-down
        SLMat3f enuRs;

        SENSOrientation::Quat ori      = _orientation->getOrientation();
        SLMat3f               rotation = SLQuat4f(ori.quatX, ori.quatY, ori.quatZ, ori.quatW).toMat3();
        enuRs.setMatrix(rotation);

        //world-yaw rotation w.r.t. world
        SLMat3f wRenu;
        wRenu.rotation(-90, 1, 0, 0);
        //combiniation of partial rotations to orientation of camera w.r.t world
        SLMat3f wRc = wRenu * enuRs * sRc;

        //camera translations w.r.t world:
        SLVec3f wtc = _userGuidanceScene->camera->updateAndGetWM().translation();

        //combination of rotation and translation:
        SLMat4f wTc;
        wTc.setRotation(wRc);
        wTc.setTranslation(wtc);

        _userGuidanceScene->camera->om(wTc);

        //ARROW ROTATION CALCULATION:
        //auto loc = _gps->getLocation();
        SENSGps::Location loc = {47.14899, 7.23354, 728.4, 1.f};

        //ROTATION OF ENU WRT. ECEF
        //calculation of ecef to world (scene) rotation matrix
        //definition of rotation matrix for ECEF to world frame rotation:
        //world frame (scene) w.r.t. ENU frame
        double phiRad = loc.latitudeDEG * Utils::DEG2RAD;  //   phi == latitude
        double lamRad = loc.longitudeDEG * Utils::DEG2RAD; //lambda == longitude
        double sinPhi = sin(phiRad);
        double cosPhi = cos(phiRad);
        double sinLam = sin(lamRad);
        double cosLam = cos(lamRad);

        SLMat3d enuRecef(-sinLam,
                         cosLam,
                         0,
                         -cosLam * sinPhi,
                         -sinLam * sinPhi,
                         cosPhi,
                         cosLam * cosPhi,
                         sinLam * cosPhi,
                         sinPhi);

        //ROTATION OF DIRECTION ARROW WRT. ENU FRAME
        //area location in ecef
        //ErlebAR::Area& area = _locations[_locId].areas[_areaId];
        SLVec3d ecefAREA;
        //schornstein brügg
        ecefAREA.lla2ecef({47.12209, 7.25821, 431.8});

        //device location in ecef
        SLVec3d ecefDEVICE;
        ecefDEVICE.lla2ecef({loc.latitudeDEG, loc.longitudeDEG, loc.altitudeM});
        //build direction vector from device to area in ecef
        SLVec3d ecefAD = ecefAREA - ecefDEVICE;
        //rotation to enu
        SLVec3d X = enuRecef * ecefAD; //enuAD
        X.normalize();
        //build y and z vectors
        SLVec3d Y;
        Y.cross(X, {0.0, 0.0, 1.0});
        Y.normalize();
        SLVec3d Z;
        Z.cross(X, Y);
        Z.normalize();
        //build rotation matrix
        // clang-format off
        SLMat3f enuRp(X.x, Y.x, Z.x,
                      X.y, Y.y, Z.y,
                      X.z, Y.z, Z.z);
        // clang-format on

        //ROTATION OF ARROW WRT. CAMERA
        SLMat3f cRw = wRc.transposed();
        SLMat3f cRp = cRw * wRenu * enuRp;
        
        //set arrow rotation
        _userGuidanceScene->updateArrowRot(cRp);
    }
    
    SENSGps* _gps = nullptr;
    SENSOrientation* _orientation = nullptr;
    //interface to update orientation of direction arrow
    UserGuidanceScene* _userGuidanceScene = nullptr;
    AreaTrackingGui* _gui = nullptr;
    const ErlebAR::Resources& _resources;
    
    //HELPER VARIABLES:
    // distance to area threshold in meter
    const float DIST_TO_AREA_THRES_M = 10.f;
    SLVec3f _areaLocation{0.f, 0.f, 0.f};
    float _areaOrientation = 0.f;
    float _currentDistanceToAreaM = 0.f;
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


