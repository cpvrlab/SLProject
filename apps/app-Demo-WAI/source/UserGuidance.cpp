#include "UserGuidance.h"
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

