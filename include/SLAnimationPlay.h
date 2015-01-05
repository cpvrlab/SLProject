//#############################################################################
//  File:      SLAnimationPlay.h
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      Autumn 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLANIMATIONPLAY_H
#define SLANIMATIONPLAY_H

#include <stdafx.h>
#include <SLEnums.h>

class SLAnimation;

//-----------------------------------------------------------------------------
//! Manages a play of an SLAnimation
/*! 
    This class manages the play states and the local time of SLAnimations.
    It manages the way the time advances and how the time loops.
    It is possible to have multiple states per animation. If we keep track
    of what nodes are affected by which SLAnimationPlay we can only manipulate
    these nodes for the time kept in the SLAnimationPlay.

    A practical example for this behaviour would be special skeleton instances
    that only keep track of SLAnimationPlay for their parent SLSkeleton.
    The skeleton instance can then change its skeletal data based on the
    states and the actual SLAnimation has to only exist once in memory.
*/
class SLAnimationPlay
{
public:
                    SLAnimationPlay (SLAnimation* parent,
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
    SLAnimation*    parentAnimation () { return _animation; }
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
    SLAnimation*    _animation;         //!< the animation this plays is referencing
    SLfloat         _localTime;         //!< the current local timestamp (eased time)
    SLfloat         _weight;            //!< the current weight
    SLfloat         _playbackRate;      //!< the current playback speed
    SLshort         _playbackDir;       //!< the current playback direction
    SLbool          _enabled;           //!< is this animation running
    SLEasingCurve   _easing;            //!< easing modifier curve (to customize start and end point easing)
    SLfloat         _linearLocalTime;   //!< linear local time used for _easing property

    SLAnimLooping   _loopingBehaviour;  //!< We support different looping behaviours
    SLbool          _gotChanged;        //!< Did this play change in the last frame
};
//-----------------------------------------------------------------------------
typedef std::map<SLstring, SLAnimationPlay*> SLMAnimationPlay;
//-----------------------------------------------------------------------------
#endif
