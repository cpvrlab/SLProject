//#############################################################################
//  File:      SLAnimPlayback.h
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      Autumn 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLANIMPLAYBACK_H
#define SLANIMPLAYBACK_H

#include <stdafx.h>
#include <SLEnums.h>

class SLAnimation;

//-----------------------------------------------------------------------------
//! Manages the playback of an SLAnimation
/*! 
This class manages the playback state and the local time of an SLAnimation.
It manages the way the time advances and how the animation loops. It has all
functionality to play, pause, stop, enable, speedup and slowdown a playback.
A list of all SLAnimPlayback is hold by the SLAnimManager.

It is possible to have multiple playbacks per animation. If we keep track
of which nodes are affected by which SLAnimPlayback we can only manipulate
these nodes for the time kept in the SLAnimPlayback.
A practical example for this behaviour would be special skeleton instances
that only keep track of SLAnimPlayback for their parent SLSkeleton.
The skeleton instance can then change its skeletal data based on the
states and the actual SLAnimation has to only exist once in memory.
*/
class SLAnimPlayback
{
public:
                    SLAnimPlayback      (SLAnimation* parent,
                                         SLfloat weight = 1.0f);

    // control functions
    void            playForward         ();
    void            playBackward        ();
    void            pause               ();
    void            skipToNextKeyframe  ();
    void            skipToPrevKeyframe  ();
    void            skipToStart         ();
    void            skipToEnd           ();

    // getters
    SLfloat         localTime           () const { return _localTime; }
    SLAnimation*    parentAnimation     () { return _animation; }
    SLfloat         playbackRate        () const { return _playbackRate; }
    SLfloat         weight              () const { return _weight; }
    SLAnimLooping   loop                () const { return _loopingBehaviour; }
    SLbool          enabled             () const { return _enabled; }
    SLEasingCurve   easing              () const { return _easing; }
    SLbool          changed             () const { return _gotChanged; }

    // setters
    void            localTime           (SLfloat time);
    void            playbackRate        (SLfloat pr) { _playbackRate = pr; }
    void            weight              (SLfloat weight) { _weight = weight; }
    void            loop                (SLAnimLooping lb) { _loopingBehaviour = lb; }
    void            enabled             (SLbool val) { _enabled = val; }
    void            easing              (SLEasingCurve ec) { _easing = ec; }
    void            changed             (SLbool changed) { _gotChanged = changed; }

    // advance time by the input real time delta
    void            advanceTime         (SLfloat delta);
    SLfloat         calcEasingTime      (SLfloat time) const;
    SLfloat         calcEasingTimeInv   (SLfloat time) const;

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
    SLbool          _gotChanged;        //!< Did this playback change in the last frame
};
//-----------------------------------------------------------------------------
typedef std::map<SLstring, SLAnimPlayback*> SLMAnimPlayback;
//-----------------------------------------------------------------------------
#endif
