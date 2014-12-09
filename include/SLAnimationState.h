//#############################################################################
//  File:      SLAnimation.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################


#ifndef SLANIMATIONSTATE_H
#define SLANIMATIONSTATE_H

#include <stdafx.h>

class SLAnimation;

/** Animationstates keep track of running animations
*/
class SLAnimationState
{
public:

    SLAnimationState(SLAnimation* parent, SLfloat weight = 1.0f);
    
    // control functions
    void            playForward();
    void            playBackward();
    void            pause();
    void            skipToNextKeyframe();
    void            skipToPrevKeyframe();
    void            skipToStart();
    void            skipToEnd();

    // getters
    SLfloat                 localTime() const { return _localTime; }
    SLAnimation*            parentAnimation() { return _parentAnim; }
    SLfloat                 playbackRate() const { return _playbackRate; }
    SLfloat                 weight() const { return _weight; }
    SLAnimLoopingBehaviour  loop() const { return _loopingBehaviour; }
    SLbool                  enabled() const { return _enabled; }
    SLEasingCurve           easing() const { return _easing; }

    // setters
    void        localTime(SLfloat time) { _localTime = time; _linearLocalTime = calcEasingTimeInv(time); }
    void        playbackRate(SLfloat pr) { _playbackRate = pr; }
    void        weight(SLfloat weight) { _weight = weight; }
    void        loop(SLAnimLoopingBehaviour lb) { _loopingBehaviour = lb; }
    void        enabled(SLbool val) { _enabled = val; }
    void        easing(SLEasingCurve ec) { _easing = ec; }

    // advance time by the input real time delta
    void        advanceTime(SLfloat delta);
    SLfloat     calcEasingTime(SLfloat time) const;
    SLfloat     calcEasingTimeInv(SLfloat time) const;

protected:
    SLAnimation*            _parentAnim;       //!< the animation this state is referencing
    SLfloat                 _localTime;
    SLfloat                 _linearLocalTime;  //!< linear local time used to be able to utilize the _easing property
    SLfloat                 _playbackRate;
    SLfloat                 _weight;
    SLbool                  _enabled;           //!< is this animation running
    SLshort                 _playbackDir;
    SLEasingCurve           _easing;            //!< easing modifier curve (to customize start and end point easing)
    SLAnimLoopingBehaviour  _loopingBehaviour;  //!< We support different looping behaviours
};

#endif