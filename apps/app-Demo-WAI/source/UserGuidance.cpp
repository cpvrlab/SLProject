#include "UserGuidance.h"

#define LOG_UG_WARN(...) Utils::log("ErlebARApp-UserGuidance", __VA_ARGS__);
#define LOG_UG_INFO(...) Utils::log("ErlebARApp-UserGuidance", __VA_ARGS__);
#define LOG_UG_DEBUG(...) Utils::log("ErlebARApp-UserGuidance", __VA_ARGS__);

UserGuidance::UserGuidance(UserGuidanceScene* userGuidanceScene,
                           AreaTrackingGui* gui,
                           SENSGps* gps,
                           SENSOrientation*
                           orientation,
                           ErlebAR::Resources& resources)
 : _userGuidanceScene(userGuidanceScene),
   _gui(gui),
   _gps(gps),
   _orientation(orientation),
   _resources(resources)
{
    reset();
}

//reset user guidance to state idle
void UserGuidance::reset()
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

void UserGuidance::areaSelected(ErlebAR::AreaId areaId, SLVec3f areaLocation, float areaOrientation)
{
    _selectedArea = areaId;
    _areaLocation = areaLocation;
    _areaOrientation = areaOrientation;
    update();
}

void UserGuidance::dataIsLoading(bool isLoading)
{
    if(isLoading != _dataIsLoading)
    {
        _dataIsLoading = isLoading;
        update();
    }
}

void UserGuidance::updateTrackingState(bool isTracking)
{
    if(isTracking != _isTracking)
    {
        _isTracking = isTracking;
        update();
    }
}

void UserGuidance::updateSensorEstimations()
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

void UserGuidance::update()
{
    //store old state
    _oldState = _state;
    //check for transitions to a new state and process the change
    while( stateTransition())
    {
        LOG_UG_DEBUG("state transition from %s to %s", mapStateToString(_oldState), mapStateToString(_state));
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
bool UserGuidance::stateTransition()
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
void UserGuidance::processStateExit()
{
    //we check the old state, that we are leaving
    switch (_oldState)
    {
        case State::IDLE: {
            break;
        }
        case State::DATA_LOADING: {
            //hide loading indicator
            _gui->hideLoading();
            break;
        }
        case State::DIR_ARROW: {
            _gui->showInfoText("");
            break;
        }
        case State::RELOCALIZING: {
            _gui->showInfoText("");
            break;
        }
        case State::RELOCALIZING_WRONG_ORIENTATION: {
            _gui->showInfoText("");
            break;
        }
        case State::TRACKING: {
            break;
        }
    }
}
//process state entry into new state
void UserGuidance::processStateEntry()
{
    switch (_state)
    {
        case State::IDLE: {
            break;
        }
        case State::DATA_LOADING: {
            //show loading indicator
            _gui->showLoading();
            break;
        }
        case State::DIR_ARROW: {
            _gui->showInfoText("follow direction arrow");
            break;
        }
        case State::RELOCALIZING: {
            _gui->showInfoText(_resources.strings().ugInfoReloc());
            break;
        }
        case State::RELOCALIZING_WRONG_ORIENTATION: {
            _gui->showInfoText("look in the correct direction");
            break;
        }
        case State::TRACKING: {
            break;
        }
    }
}

//update the current state
void UserGuidance::processState()
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

void UserGuidance::estimateArrowOrientation()
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

/*
#include <Gui/AreaTrackingGui.h>

UserGuidance::UserGuidance(AreaTrackingGui * gui)
{
    _gui = gui;
    _lastWAIState = WAI::TrackingState_None;
    _timer.start();

    _alignImgInfo.updateFct = [](float tStart, float tTerminate, AreaTrackingGui* gui, bool &terminate)
    {
        bool lookingToward = false;
        if (lookingToward)
            terminate = true;

        float alpha = 0.8 - tTerminate * 0.8;
        if (alpha >= 0)
        {
            gui->showText("Align the camera view to this image");
            gui->showImageAlignTexture(alpha);
            return true;
        }
        else
            gui->showImageAlignTexture(0.f);
        return false;
    };

    _moveLeftRight.updateFct = [](float tStart, float tTerminate, AreaTrackingGui* gui, bool &terminate)
    {
        bool relocalize = false;
        if (relocalize || terminate)
            return false;

        gui->showText("Move left and right");
        return true;
    };

    _trackingMarker.updateFct = [](float tStart, float tTerminate, AreaTrackingGui* gui, bool &terminate)
    {
        //float distanceToMarker;
        //SLVec3 dirToMarker(0, 0, 0);
        if (terminate)
            return false;

        if (tStart < 3)
            gui->showText("Move slowly backward");
        else if (tStart < 6)
            gui->showText("Align the view to the augmentation");
        else if (tStart < 8)
            gui->showText("Keep the marker in view");
        else
            return false;
        return true;
    };

    _trackingStarted.updateFct = [](float tStart, float tTerminate, AreaTrackingGui* gui, bool &terminate)
    {
        if (terminate)
            return false;
        if (tStart < 2)
            gui->showText("You can now look through different angle");
        else if (tStart < 4)
            gui->showText("Move slowly");
        else
            return false;
        return true;
    };
}

void UserGuidance::flush()
{
    if (_queuedInfos.empty())
        return;
    UserGuidanceInfo* info = _queuedInfos.front();
    info->terminate();
    while(!_queuedInfos.empty())
        _queuedInfos.pop();

    _queuedInfos.push(info);
}

void UserGuidance::update(WAI::TrackingState state)
{
    if (state != _lastWAIState)
    {
        _lastWAIState = state;
        flush();

        if (state == WAI::TrackingState_TrackingLost)
        {
            _queuedInfos.push(&_alignImgInfo);
            _queuedInfos.push(&_moveLeftRight);
        }
        else if (state == WAI::TrackingState_TrackingStart || state == WAI::TrackingState_TrackingOK)
        {
            if (_marker)
            {
                _queuedInfos.push(&_trackingMarker);
                _queuedInfos.push(&_trackingStarted);
            }
            else
                _queuedInfos.push(&_trackingStarted);
        }
    }

    if (_queuedInfos.empty())
    {
        _gui->showText("");
        return;
    }

    float t = _timer.elapsedTimeInSec();

    UserGuidanceInfo* info = _queuedInfos.front();
    if (!info->update(t, _gui))
        _queuedInfos.pop();
}

*/
