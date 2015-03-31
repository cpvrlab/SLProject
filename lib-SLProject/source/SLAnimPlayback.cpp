//#############################################################################
//  File:      SLAnimPlay.cpp
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif
#include <SLAnimation.h>
#include <SLAnimPlayback.h>

//-----------------------------------------------------------------------------
/*! Constructor
*/
SLAnimPlayback::SLAnimPlayback(SLAnimation* parent, SLfloat weight)
                : _animation(parent),
                _localTime(0.0f),
                _linearLocalTime(0.0f),
                _playbackRate(1.0f),
                _playbackDir(1),
                _weight(weight),
                _enabled(false),
                _easing(EC_linear),
                _loopingBehaviour(AL_loop)
{
}
//-----------------------------------------------------------------------------
/*! Advances the time of the playbackplay based on its different easing parameters.
*/
void SLAnimPlayback::advanceTime(SLfloat delta)
{
    if (!_enabled)
        return;

    if (delta == 0.0f)
        return;

    // preserve time before update
    SLfloat prevTime = _linearLocalTime;

    _linearLocalTime += delta * _playbackRate * _playbackDir;
    
    // fix invalid inputs
    if (_linearLocalTime > _animation->lengthSec())
    {
        // wrap around on loop, else just stay on last frame
        switch (_loopingBehaviour)
        {   case AL_once:          _linearLocalTime = _animation->lengthSec(); _enabled = false; break;
            case AL_loop:          _linearLocalTime = 0.0f; break;
            case AL_pingPong:      _linearLocalTime = _animation->lengthSec(); _playbackDir *= -1; break;
            case AL_pingPongLoop:  _linearLocalTime = _animation->lengthSec(); _playbackDir *= -1; break;
        }
    }
    // fix negative inputs, playback rate could be negative
    else if (_linearLocalTime < 0.0f)
    {
        while (_linearLocalTime < 0.0f)
            _linearLocalTime += _animation->lengthSec();

        switch (_loopingBehaviour)
        {   case AL_once:          _linearLocalTime = 0.0f; _enabled = false; break;
            case AL_loop:          _linearLocalTime = _animation->lengthSec(); break;
            case AL_pingPong:      _linearLocalTime = 0.0f; _enabled = false; break; // at the moment pingPong stops when reaching 0, if we start with a reverse direction this is illogical
            case AL_pingPongLoop:  _linearLocalTime = 0.0f; _playbackDir *= -1; break;
        }
    }     

    // don't go any further, nothing's changed
    if (_linearLocalTime == prevTime)
        return;

    // mark the playback as changed
    _gotChanged = true;

    // update the final eased local time
    _localTime = calcEasingTime(_linearLocalTime);
}

//-----------------------------------------------------------------------------
/*! Set this playback to be playing forward.
*/
void SLAnimPlayback::playForward()
{
    _enabled = true;
    _playbackDir = 1;
}

//-----------------------------------------------------------------------------
/*! Set this playback to be playing backward.
*/
void SLAnimPlayback::playBackward()
{
    _enabled = true;
    _playbackDir = -1;
}

//-----------------------------------------------------------------------------
/*! Set this playback to be paused.
*/
void SLAnimPlayback::pause()
{
    // a paused anmation is an enabled animation that has a 0 direction multiplier
    _enabled = true;
    _playbackDir = 0;
}

//-----------------------------------------------------------------------------
/*! Set the local time of this playback to be on the time of the next keyframe.
*/
void SLAnimPlayback::skipToNextKeyframe()
{
    SLfloat time = _animation->nextKeyframeTime(_localTime);
    localTime(time);
}

//-----------------------------------------------------------------------------
/*! Set the local time of this playback to be on the time of the previous keyframe.
*/
void SLAnimPlayback::skipToPrevKeyframe()
{
    SLfloat time = _animation->prevKeyframeTime(_localTime);
    localTime(time);
}


//-----------------------------------------------------------------------------
/*! Set the local time of this playback to the starting time.
*/
void SLAnimPlayback::skipToStart()
{
    localTime(0.0f);
}

//-----------------------------------------------------------------------------
/*! Set the local time of this animation to the end time.
*/
void SLAnimPlayback::skipToEnd()
{
    localTime(_animation->lengthSec());
}

//-----------------------------------------------------------------------------
/*! Setter for the local time parameter. Takes the currently active easing
curve into consideration.
*/
void SLAnimPlayback::localTime(SLfloat time)
{
    // Set the eased time
    _localTime = time; 

    // calculate the equivalent linear time from the new eased time
    _linearLocalTime = calcEasingTimeInv(time);

    // mark changed
    _gotChanged = true;
}


//-----------------------------------------------------------------------------
//! Applies the easing time curve to the input time.
/*! See also the declaration of the SLEasingCurve enumeration for the different
easing curve type that are taken from Qt QAnimation and QEasingCurve class. 
See http://qt-project.org/doc/qt-4.8/qeasingcurve.html#Type-enum
*/
SLfloat SLAnimPlayback::calcEasingTime(SLfloat time) const
{
    SLfloat x = time / _animation->lengthSec();
    SLfloat y = 0.0f;

    switch (_easing)
    {
        case EC_linear:      y = x; break;

        case EC_inQuad:      y = pow(x, 2.0f); break;
        case EC_outQuad:     y = -pow(x - 1.0f, 2.0f) + 1.0f; break;
        case EC_inOutQuad:   y = (x < 0.5f) ? 2.0f * pow(x, 2.0f) : -2.0f * pow(x - 1.0f, 2.0f) + 1.0f; break;
        case EC_outInQuad:   y = (x < 0.5f) ?-2.0f * pow(x - 0.5f, 2.0f) + 0.5f : 2.0f*pow(x - 0.5f, 2.0f) + 0.5f; break;
   
        case EC_inCubic:     y = pow(x, 3.0f); break;
        case EC_outCubic:    y = pow(x - 1.0f, 3.0f) + 1.0f; break;
        case EC_inOutCubic:  y = (x < 0.5f) ? 4.0f * pow(x, 3.0f) : 4.0f * pow(x - 1.0f, 3.0f) + 1.0f; break;
        case EC_outInCubic:  y = 4.0f*pow(x-0.5f,3.0f) + 0.5f; break;

        case EC_inQuart:     y = pow(x, 4.0f); break;
        case EC_outQuart:    y = -pow(x - 1.0f, 4.0f) + 1.0f; break;
        case EC_inOutQuart:  y = (x < 0.5f) ? 8.0f * pow(x, 4.0f) : -8.0f * pow(x - 1.0f, 4.0f) + 1.0f; break;
        case EC_outInQuart:  y = (x < 0.5f) ? -8.0f * pow(x - 0.5f, 4.0f) + 0.5f : 8.0f * pow(x - 0.5f, 4.0f) + 0.5f; break;
   
        case EC_inQuint:     y = pow(x, 5.0f); break;
        case EC_outQuint:    y = pow(x - 1.0f, 5.0f) + 1.0f; break;
        case EC_inOutQuint:  y = (x < 0.5f) ? 16.0f * pow(x, 5.0f) : 16.0f*pow(x - 1.0f, 5.0f) + 1.0f; break;
        case EC_outInQuint:  y = 16.0f*pow(x-0.5f,5.0f) + 0.5f; break; 

        case EC_inSine:      y = sin(x*SL_PI*0.5f- SL_PI*0.5f)+ 1.0f; break;
        case EC_outSine:     y = sin(x*SL_PI*0.5f); break;
        case EC_inOutSine:   y = 0.5f*sin(x*SL_PI - SL_PI*0.5f) + 0.5f; break;
        case EC_outInSine:   y = (x<0.5f) ? 0.5f*sin(x*SL_PI) : 0.5f*sin(x*SL_PI - SL_PI) + 1.0f; break;    

        default: y = x; 
    }
    
    return y * _animation->lengthSec();
}

//-----------------------------------------------------------------------------
/*! Inverse functions for the above easing curve functions.
*/
SLfloat SLAnimPlayback::calcEasingTimeInv(SLfloat time) const
{
    SLfloat x = time / _animation->lengthSec();
    SLfloat y = 0.0f;

    switch (_easing)
    {
        case EC_linear:      y = x; break;

        case EC_inQuad:      y = sqrt(x); break;
        case EC_outQuad:     y = 1.0f - sqrt(1.0f - x); break;
        case EC_inOutQuad:   y = (x < 0.5f) ? sqrt(x) / sqrt(2.0f) : 
                                              1.0f - sqrt(1.0f - x) / sqrt(2.0f); break;
        case EC_outInQuad:   y = (x < 0.5f) ? 0.5f - 0.25f * sqrt(4.0f - 8.0f * x) : 
                                              0.5f + 0.25f * sqrt(8.0f * x - 4.0f); break;
            
        case EC_inCubic:     y = pow(x, 1.0f / 3.0f); break;
        case EC_outCubic:    y = 1.0f - pow(1.0f - x, 1.0f / 3.0f); break;
        case EC_inOutCubic:  y = (x < 0.5f) ?  pow(x, 1.0f / 3.0f) / pow(4.0f, 1.0f / 3.0f) : 
                                               1.0f - pow(1.0f - x, 1.0f / 3.0f) / pow(4.0f, 1.0f/3.0f); break;
        case EC_outInCubic:  y = (x < 0.5f) ? -pow((0.5f-x) / 4.0f, 1.0f / 3.0f) + 0.5f : 
                                               pow((x - 0.5f) / 4.0f, 1.0f / 3.0f) + 0.5f; break;
            
        case EC_inQuart:     y = pow(x, 1.0f / 4.0f); break;
        case EC_outQuart:    y = 1.0f - pow(1.0f - x, 1.0f / 4.0f); break;
        case EC_inOutQuart:  y = (x < 0.5f) ?  pow(x, 1.0f / 4.0f) / pow(8.0f, 1.0f/4.0f) : 
                                               1.0f - pow(1.0f - x, 1.0f / 4.0f) / pow(8.0f, 1.0f / 4.0f); break;
        case EC_outInQuart:  y = (x < 0.5f) ? -pow(0.5f - x, 1.0f / 4.0f) / pow(8.0f, 1.0f / 4.0f) + 0.5f :
                                               pow(x - 0.5f, 1.0f / 4.0f) / pow(8.0f, 1.0f / 4.0f) + 0.5f; break;  

        case EC_inQuint:     y = pow(x, 1.0f/5.0f); break;
        case EC_outQuint:    y = 1.0f - pow(1.0f - x, 1.0f/5.0f); break;
        case EC_inOutQuint:  y = (x < 0.5f) ?  pow(x, 1.0f / 5.0f) / pow(16.0f, 1.0f / 5.0f) :
                                               1.0f - pow(1.0f-x, 1.0f / 5.0f) / pow(16.0f, 1.0f / 5.0f); break;
        case EC_outInQuint:  y = (x < 0.5f) ? -pow(0.5f - x, 1.0f / 5.0f)/pow(16.0f, 1.0f / 5.0f) + 0.5f : 
                                               pow(x - 0.5f, 1.0f / 5.0f) / pow(16.0f, 1.0f / 5.0f) + 0.5f; break;

        case EC_inSine:      y = -2.0f * asin(1.0f-x) / SL_PI + 1.0f; break;
        case EC_outSine:     y = -2.0f * acos(x) / SL_PI + 1.0f; break;
        case EC_inOutSine:   y = acos(1.0f - 2.0f * x) / SL_PI; break;
        case EC_outInSine:   y = (x < 0.5f) ? asin(2.0f * x         ) / SL_PI : 
                                              asin(2.0f * (x - 1.0f)) / SL_PI + 1.0f; break;                                  
        default: y = x; 
    }

    return y * _animation->lengthSec();
}
//-----------------------------------------------------------------------------
