#ifndef ERLEBAR_USERGUIDANCE_H
#define ERLEBAR_USERGUIDANCE_H

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

    bool _marker;
    bool _trackedOnce;
};

#endif
