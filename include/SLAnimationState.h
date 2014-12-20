//#############################################################################
//  File:      SLAnimationState.h
//  Author:    Marc Wacker
//  Date:      Autumn 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLANIMATIONSTATE_H
#define SLANIMATIONSTATE_H

#include <stdafx.h>
#include <SLEnums.h>

class SLAnimation;

//-----------------------------------------------------------------------------
//! Animationstates keep track of running animations
class SLAnimationState
{
public:
                    SLAnimationState(SLAnimation* parent,
                                     SLfloat weight = 1.0f);
    
    // control functions
    void            playForward     ();
    void            playBackward    ();
    void            pause           ();
    void            skipToNextKeyframe();
    void            skipToPrevKeyframe();
    void            skipToStart     ();
    void            skipToEnd       ();

    // getters
    SLfloat         localTime       () const { return _localTime; }
    SLAnimation*    parentAnimation () { return _parentAnim; }
    SLfloat         playbackRate    () const { return _playbackRate; }
    SLfloat         weight          () const { return _weight; }
    SLAnimLooping   loop            () const { return _loopingBehaviour; }
    SLbool          enabled         () const { return _enabled; }
    SLEasingCurve   easing          () const { return _easing; }
    SLbool          changed         () const { return _gotChanged; }

    // setters
    void            localTime       (SLfloat time);
    void            playbackRate    (SLfloat pr) { _playbackRate = pr; }
    void            weight          (SLfloat weight) { _weight = weight; }
    void            loop            (SLAnimLooping lb) { _loopingBehaviour = lb; }
    void            enabled         (SLbool val) { _enabled = val; }
    void            easing          (SLEasingCurve ec) { _easing = ec; }
    void            changed         (SLbool changed) { _gotChanged = changed; }

    // advance time by the input real time delta
    void            advanceTime     (SLfloat delta);
    SLfloat         calcEasingTime  (SLfloat time) const;
    SLfloat         calcEasingTimeInv(SLfloat time) const;

protected:
    SLAnimation*    _parentAnim;        //!< the animation this state is referencing
    SLfloat         _localTime;         //!< ???
    SLfloat         _linearLocalTime;   //!< linear local time used f _easing property
    SLfloat         _playbackRate;      //!< ???
    SLfloat         _weight;            //!< ???
    SLbool          _enabled;           //!< is this animation running
    SLshort         _playbackDir;       //!< ???
    SLEasingCurve   _easing;            //!< easing modifier curve (to customize start and end point easing)
    SLAnimLooping   _loopingBehaviour;  //!< We support different looping behaviours
    SLbool          _gotChanged;        //!< Did this state change in the last frame
};
//-----------------------------------------------------------------------------
typedef std::map<SLstring, SLAnimationState*> SLMAnimationState;
//-----------------------------------------------------------------------------
#endif
