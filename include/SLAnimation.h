//#############################################################################
//  File:      SLAnimation.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLANIMATION_H
#define SLANIMATION_H

#include <stdafx.h>
#include <SLEnums.h>
#include <SLKeyframe.h>

class SLNode;
class SLCurve;

//-----------------------------------------------------------------------------
//! SLAnimation implements simple keyframe animation.
/*!
An animation consists out of at least two keyframes. A keyframe is defines 
position (translation), rotation and scaling for a specific time point with in 
an animation. An animation interpolates between two keyframes the position, 
rotation and scaling.
The first keyframe's time will allways be 0. The _totalTime of an animation in
seconds is the sum of all keyframe times.
The animation is applied to the owner shapes local transform matrix in 
animate(SLNode* node, SLfloat elapsedTimeMS) proportional to the elapsed
time of the frame.
The animation mode [once, loop, pingPong, pingPongLoop] determines what happens
if the total animation time is over.
Because the transform matrices of the owner node is modified during animation,
the animation instance keep the original owner matrices in _om and _wm.
The animation easing determines how constant the velocity of the animation. It
is defined by an easing curve that is by default set to a linear motion with 
a constant speed. See the declaration of SLEasingCurve for all possible easing
curves.
*/
class SLAnimation : SLObject
{
   public:           //! Multiple keyframes animation ctor
                     SLAnimation (SLVKeyframe keyframes,
                                  SLVec3f* controls = 0,
                                  SLAnimMode mode = once, 
                                  SLEasingCurve easing = linear,
                                  SLstring name="myKeyframesAnimation");
                                  
                     //! Single keyframe animation ctor
                     SLAnimation (SLKeyframe keyframe,
                                  SLAnimMode mode = once,
                                  SLEasingCurve easing = linear, 
                                  SLstring name="myKeyframeAnimation");

                     //! Single translation animation
                     SLAnimation (SLfloat time,
                                  SLVec3f translation,
                                  SLAnimMode mode = once,
                                  SLEasingCurve easing = linear, 
                                  SLstring name="myTranslationAnimation");
                     
                     //! Single rotation animation
                     SLAnimation (SLfloat time,
                                  SLfloat angleDEG,
                                  SLVec3f rotationAxis,
                                  SLAnimMode mode = once,
                                  SLEasingCurve easing = linear,
                                  SLstring name="myRotationAnimation");
                     
                     //! Single scaling animation
                     SLAnimation (SLfloat time,
                                  SLfloat scaleX,
                                  SLfloat scaleY,
                                  SLfloat scaleZ,
                                  SLAnimMode mode = once,
                                  SLEasingCurve easing = linear,
                                  SLstring name="myScalingAnimation");
                     
                     //! Elliptic animation with 2 radiis on 2 axis
                     SLAnimation (SLfloat time,
                                  SLfloat radiusA, SLAxis axisA,
                                  SLfloat radiusB, SLAxis axisB,
                                  SLAnimMode mode = once,
                                  SLEasingCurve easing = linear,
                                  SLstring name="myEllipticAnimation");

                     //!Copy constructor
                     SLAnimation (SLAnimation& anim);

                    ~SLAnimation ();

      void           init        (SLVec3f* controls);
      void           animate     (SLNode* node, SLfloat elapsedTimeMS);
      void           drawWS      ();

      SLfloat        totalTime   () {return _totalTime;}
      SLbool         isFinished  () {return _isFinished;}

   private:
      SLfloat        easing      (SLfloat time);

      SLVKeyframe    _keyframes;    //!< Vector with keyframes
      SLCurve*       _curve;        //!< Interpolation curve for translation
      SLfloat        _totalTime;    //!< Total cummultated time in seconds 
      SLfloat        _currentTime;  //!< Current time in seconds during anim.
      SLbool         _isFinished;   //!< Flag if animation is finished
      SLAnimMode     _mode;         //!< Animation mode
      SLEasingCurve  _easing;       //!< Easing curve type
      SLfloat        _direction;    //!< Direction 1=forewards -1=backwards
      SLMat4f        _om;           //!< Original object transform matrix
      SLMat4f        _wm;           //!< Original world transform matrix
};
//-----------------------------------------------------------------------------
#endif
