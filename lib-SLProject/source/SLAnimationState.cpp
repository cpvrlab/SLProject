
#include <stdafx.h>
#include "SLAnimation.h"
#include "SLAnimationState.h"

SLAnimationState::SLAnimationState(SLAnimation* parent, SLfloat weight)
: _parentAnim(parent), 
_localTime(0.0f),
_playbackRate(1.0f),
_weight(weight),
_loop(true),
_enabled(false)
{ }

void SLAnimationState::advanceTime(SLfloat delta)
{
    if (!_enabled)
        return;

    _localTime += delta * _playbackRate;

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