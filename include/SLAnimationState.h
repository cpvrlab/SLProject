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

    SLfloat     localTime() const { return _localTime; }
    SLAnimation* parentAnimation() { return _parentAnim; }
    SLfloat     playbackRate() const { return _playbackRate; }
    SLfloat     weight() const { return _weight; }
    SLbool      loop() const { return _loop; }
    SLbool      enabled() const { return _enabled; }

    void        playbackRate(SLfloat pr) { _playbackRate = pr; }
    void        weight(SLfloat weight) { _weight = weight; }
    void        loop(SLbool val) { _loop = val; }
    void        enabled(SLbool val) { _enabled = val; }

    void        advanceTime(SLfloat delta);

protected:
    SLAnimation*    _parentAnim;       //!< the animation this state is referencing
    SLfloat	        _localTime;
    SLfloat	        _playbackRate;
    SLfloat         _weight;
    SLbool          _loop;              //!< is this animation looping
    SLbool          _enabled;           //!< is this animation running
};


#endif