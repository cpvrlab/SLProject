
#include <stdafx.h>
#include "SLAnimation.h"
#include "SLAnimationState.h"

SLAnimationState::SLAnimationState(SLAnimation* parent, SLfloat weight)
: _parentAnim(parent), 
_localTime(0.0f),
_playbackRate(1.0f),
_playbackDir(1),
_weight(weight),
_loop(true),
_enabled(false)
{ }

void SLAnimationState::advanceTime(SLfloat delta)
{
    if (!_enabled)
        return;

    _localTime += delta * _playbackRate * _playbackDir;

    // fix invalid inputs
    if (_localTime > _parentAnim->length())
    {
        // wrap around on loop, else just stay on last frame
        if (_loop)
            _localTime = fmod(_localTime, _parentAnim->length());
        else
            _localTime = _parentAnim->length();
    }
    // fix negative inputs, playback rate could be negative
    else if (_localTime < 0.0f)
    {
        if (_loop)
            _localTime = fmod(_localTime, _parentAnim->length()) + _parentAnim->length();
        else
            _localTime = 0.0f;
    }     
}



void SLAnimationState::playForward()
{
    _enabled = true;
    _playbackDir = 1;
}
void SLAnimationState::playBackward()
{
    _enabled = true;
    _playbackDir = -1;
}
void SLAnimationState::pause()
{
    // @todo is a paused animation disabled OR is it enabled but just not advancing time?
    //       currently we set the direction multiplier to 0
    _enabled = true;
    _playbackDir = 0;
}
void SLAnimationState::skipToNextKeyframe()
{
    SLfloat time = _parentAnim->nextKeyframeTime(_localTime);
    localTime(time);
}
void SLAnimationState::skipToPrevKeyframe()
{
    SLfloat time = _parentAnim->prevKeyframeTime(_localTime);
    localTime(time);
}
void SLAnimationState::skipToStart()
{
    localTime(0.0f);
}
void SLAnimationState::skipToEnd()
{
    localTime(_parentAnim->length());
}