//#############################################################################
//  File:      SLAnimation.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLAnimation.h>
#include <SLNode.h>
#include <SLCurveBezier.h>

//-----------------------------------------------------------------------------
/*! 
Animation ctor for multiple keyframe animation where the keyframe translation
positions will define a Bézier curve. The control points of the Bézier curve
will be automatically calculated if controls a zero pointer. If an array of 
control points is passed it must have the size of 2*(numKeyframes-1).
*/
SLAnimation::SLAnimation(SLVKeyframe keyframes,
                         SLVec3f* controls,
                         SLAnimMode mode,
                         SLEasingCurve easing,
                         SLstring name) : SLObject(name)
{
    _keyframes = keyframes;
    _mode = mode;
    _easing = easing;
    _curve = 0;
    init(controls);
}
//-----------------------------------------------------------------------------
/*! 
Animation ctor for single keyframe animation. The animation will interpolate
between 2 keyframes: An initial keyframe without any transform at time zero and
the passed keyframe transform at time passed.
*/
SLAnimation::SLAnimation(SLKeyframe keyframe,
                         SLAnimMode mode, 
                         SLEasingCurve easing, 
                         SLstring name) : SLObject(name)
{  
    _keyframes.push_back(SLKeyframe());
    _keyframes.push_back(keyframe);
    _mode = mode;
    _easing = easing;
    _curve = 0;
    init(0);
}
//-----------------------------------------------------------------------------
/*! 
Animation ctor for single translation animation. The animation will interpolate
between 2 keyframes: An initial keyframe without any transform at time zero and
the translation transform at time passed.
*/
SLAnimation::SLAnimation(SLfloat time, 
                         SLVec3f position,
                         SLAnimMode mode,
                         SLEasingCurve easing,
                         SLstring name) : SLObject(name)
{
    _keyframes.push_back(SLKeyframe());
    _keyframes.push_back(SLKeyframe(time, position));
    _mode = mode;
    _easing = easing;
    _curve = 0;
    init(0);
}
//-----------------------------------------------------------------------------
/*! 
Animation ctor for single rotation animation. The animation will interpolate
between 2 keyframes: An initial keyframe without any transform at time zero and
the rotation transform at time passed.
*/
SLAnimation::SLAnimation(SLfloat time, 
                         SLfloat rotAngleDEG, 
                         SLVec3f rotAxis, 
                         SLAnimMode mode,
                         SLEasingCurve easing,
                         SLstring name) : SLObject(name)
{
    _keyframes.push_back(SLKeyframe());
    _keyframes.push_back(SLKeyframe(time, 
                                    rotAngleDEG, 
                                    rotAxis));
    _mode = mode;
    _easing = easing;
    _curve = 0;
    init(0);
}
//-----------------------------------------------------------------------------
/*! 
Animation ctor for single scaling animation. The animation will interpolate
between 2 keyframes: An initial keyframe without any transform at time zero and
the scaling transform at time passed.
*/
SLAnimation::SLAnimation(SLfloat time, 
                         SLfloat scaleX, 
                         SLfloat scaleY,
                         SLfloat scaleZ, 
                         SLAnimMode mode,
                         SLEasingCurve easing,
                         SLstring name) : SLObject(name)
{
    _keyframes.push_back(SLKeyframe());
    _keyframes.push_back(SLKeyframe(time, 
                                    scaleX, scaleY, scaleZ));
    _mode = mode;
    _easing = easing;
    _curve = 0;
    init(0);
}
//-----------------------------------------------------------------------------
/*! 
Elliptic animation in a plane defined by 2 radiis along the coordinate axisA 
and axisB. 
The control points to approximate an ellipse or a circle with a Bézier curve
use the magic factor kappa. 
See: http://www.whizkidtech.redprince.net/bezier/circle/kappa/
*/
SLAnimation::SLAnimation(SLfloat time,
                         SLfloat radiusA, SLAxis axisA,
                         SLfloat radiusB, SLAxis axisB,
                         SLAnimMode mode,
                         SLEasingCurve easing,
                         SLstring name) : SLObject(name)
{  
    assert(axisA!=axisB && radiusA>0 && radiusB>0);

    /* The ellipse is defined by 5 keyframes: A,B,C,D and again A

        c2----B----c1

    c3                 c0
    ¦                   ¦
    ¦         ¦         ¦
    C       --0--       A
    ¦         ¦         ¦
    ¦                   ¦
    c4                 c7 

        c5----D----c6
    */

    SLVec3f A(0,0,0); A.comp[axisA] =  radiusA;
    SLVec3f B(0,0,0); B.comp[axisB] =  radiusB;
    SLVec3f C(0,0,0); C.comp[axisA] = -radiusA;
    SLVec3f D(0,0,0); D.comp[axisB] = -radiusB;

    // Control points with the magic factor kappa for control points
    SLfloat k = 0.5522847498f;

    SLVec3f controls[8];
    for (SLint i=0; i<8; ++i) controls[i].set(0,0,0);
    controls[0].comp[axisA] = radiusA; controls[0].comp[axisB] = k *  radiusB;
    controls[1].comp[axisB] = radiusB; controls[1].comp[axisA] = k *  radiusA;
    controls[2].comp[axisB] = radiusB; controls[2].comp[axisA] = k * -radiusA;
    controls[3].comp[axisA] =-radiusA; controls[3].comp[axisB] = k *  radiusB;
    controls[4].comp[axisA] =-radiusA; controls[4].comp[axisB] = k * -radiusB;
    controls[5].comp[axisB] =-radiusB; controls[5].comp[axisA] = k * -radiusA;
    controls[6].comp[axisB] =-radiusB; controls[6].comp[axisA] = k *  radiusA;
    controls[7].comp[axisA] = radiusA; controls[7].comp[axisB] = k * -radiusB;

    // Add keyframes
    SLfloat t4 = time / 4.0f;
    _keyframes.push_back(SLKeyframe(0,  A));
    _keyframes.push_back(SLKeyframe(t4, B));
    _keyframes.push_back(SLKeyframe(t4, C));
    _keyframes.push_back(SLKeyframe(t4, D));
    _keyframes.push_back(SLKeyframe(t4, A));
   
    _curve = 0;
    _mode = mode;
    _easing = easing;
    init(controls);
}
//-----------------------------------------------------------------------------
SLAnimation::SLAnimation(SLAnimation& anim)
{
    _keyframes = anim._keyframes;
    _curve = 0;
    _totalTime = anim._totalTime;
    _currentTime = anim._currentTime;
    _direction = anim._direction;
    _isFinished = anim._isFinished;
    _mode = anim._mode;
    _easing = anim._easing;
    _wm = anim._wm;
    _om = anim._om;
    init(0);
}
//-----------------------------------------------------------------------------
SLAnimation::~SLAnimation()
{
    if (_curve) delete _curve;
}
//-----------------------------------------------------------------------------
/*!
SLAnimation::init loops through all keyframes and determines the total 
animation time. If more than two keyframes exist a Bézier curve is created for
the position interpolation.
*/
void SLAnimation::init(SLVec3f* controls)
{
    // Cummulate total time
    _totalTime = 0;
    for (SLint i=0; i<_keyframes.size(); ++i)
        _totalTime += _keyframes[i].time();

    if (_keyframes.size() > 1)
    {
        if (_curve) delete _curve;

        // Build curve data w. cummulated times
        SLVec3f* points = new SLVec3f[_keyframes.size()];
        SLfloat* times  = new SLfloat[_keyframes.size()];
        SLfloat  curTime = 0;
        for (SLint i=0; i<_keyframes.size(); ++i)
        {   curTime += _keyframes[i].time();
            points[i] = _keyframes[i].position();
            times[i] = curTime;
        }

        // create curve and delete temp arrays again
        _curve = new SLCurveBezier(points, times, (SLint)_keyframes.size(), controls);
        delete[] points;
        delete[] times;
    }

    _currentTime =-1.0f;
    _direction   = 1.0f;
    _isFinished  = false;
}
//-----------------------------------------------------------------------------
/*!
SLAnimation::animate does the transformation changes by updating the local
transform matrix of the passed node. The shapes matrix is updated in 
fixed order: Translation, rotation and scale. 
*/
void SLAnimation::animate(SLNode* node, SLfloat elapsedTimeMS)
{
    if (_isFinished) return;

    // Get the original object transform once
    if (_currentTime < 0)
    {   _om = node->om();
        //_wm = node->wm();
        _currentTime = 0.0f;
    }

    // Apply the easing curve to the current time
    SLfloat easingTime = easing(_currentTime);

    // Determine involved neighbouring keyframes
    SLKeyframe *k1=0, *k2=0;   // Pointers to previous and next keyframe   
    SLfloat t1 = 0.0f, t2;     // Time of previous and next keyframe
    SLfloat frac;              // fraction of next keyframe (0-1)

    for (SLint i=0; i<_keyframes.size()-1; ++i)
    {  
        // Cummulated time at last frame
        t1 += _keyframes[i].time();
      
        if (easingTime >= t1 && easingTime <=  t1 + _keyframes[i+1].time())
        {  k1 = &_keyframes[i];
            k2 = &_keyframes[i+1];
            t2 = t1 + k2->time();
            frac = (easingTime - t1) / (t2-t1);
            break;
        }
    }

    if (k1 && k2)
    {
        // Reset original object transform
        node->om(_om);

        // Add translation if the keyframes positions are different
        if (k1->position() != k2->position())
            node->translate(_curve->evaluate(easingTime), TS_Local);
   
        // Add rotation if the keyframe rotations are different
        if (k1->orientation() != k2->orientation())
        {  
            // if previous rotation is a zero rotation do an angle interpolation
            if (k1->orientation()==SLQuat4f(0,0,0,1))
            {  SLfloat angleDEG;
            SLVec3f axis;
            k2->orientation().toAngleAxis(angleDEG, axis);
            node->rotate(frac * angleDEG, axis);
            }
            else // otherwise do quaternion interpolation
                node->rotate(k1->orientation().lerp(k2->orientation(), frac)); // node->multiply(k1->orientation().lerp(k2->orientation(), frac).toMat4());
        }
        // Add scaling if the keyframe scalings are different
        if (k1->scaling() != k2->scaling())
            node->scale((1-frac)*k1->scaling() + frac*k2->scaling());
      
        // increment or decrement time
        _currentTime += elapsedTimeMS / 1000.0f * _direction;
    }

    //cout << _currentTime << endl;

    // Handle the animation mode
    if (_currentTime > _totalTime)
    {  
        switch (_mode)
        {   case once:
                _currentTime = 0.0f;
                _isFinished = true;
                break;
            case loop:
                _currentTime = 0.0f;
                break;
            case pingPong:
            case pingPongLoop:
                _currentTime = _totalTime;
                _direction = -1.0f;
                break;
            default: 
                _isFinished = true;
                break;
        }
    } 
    else
    if (_currentTime < 0.0f)
    {
        _currentTime = 0.0f;
        _direction = 1.0f;      
        if (_mode == pingPong)
            _isFinished = true;
    }
}
//-----------------------------------------------------------------------------
/*!
SLAnimation::draw draws the animation curve in world space if the AABB of the
animation owner are drawn.
*/
void SLAnimation::drawWS()
{   
    if (_keyframes.size() > 1 && _curve)
        _curve->draw(_wm);
}
//-----------------------------------------------------------------------------
//! Applies the easing time curve to the input time.
/*! See also the declaration of the SLEasingCurve enumeration for the different
easing curve type that are taken from Qt QAnimation and QEasingCurve class. 
See http://qt-project.org/doc/qt-4.8/qeasingcurve.html#Type-enum
*/
SLfloat SLAnimation::easing(SLfloat time)
{
    SLfloat x = time / _totalTime;
    SLfloat y = 0.0f;

    switch (_easing)
    {
        case linear:      y = x; break;

        case inQuad:      y =                  pow(x     ,2.0f);        break;
        case outQuad:     y =                 -pow(x-1.0f,2.0f) + 1.0f; break;
        case inOutQuad:   y =  (x<0.5f) ? 2.0f*pow(x     ,2.0f) : 
                                        -2.0f*pow(x-1.0f,2.0f) + 1.0f; break;
        case outInQuad:   y =  (x<0.5f) ?-2.0f*pow(x-0.5f,2.0f) + 0.5f : 
                                          2.0f*pow(x-0.5f,2.0f) + 0.5f; break;
   
        case inCubic:     y =                  pow(x     ,3.0f);        break;
        case outCubic:    y =                  pow(x-1.0f,3.0f) + 1.0f; break;
        case inOutCubic:  y =  (x<0.5f) ? 4.0f*pow(x     ,3.0f) : 
                                          4.0f*pow(x-1.0f,3.0f) + 1.0f; break;
        case outInCubic:  y =             4.0f*pow(x-0.5f,3.0f) + 0.5f; break;

        case inQuart:     y =                  pow(x     ,4.0f);        break;
        case outQuart:    y =                 -pow(x-1.0f,4.0f) + 1.0f; break;
        case inOutQuart:  y =  (x<0.5f) ? 8.0f*pow(x     ,4.0f) : 
                                         -8.0f*pow(x-1.0f,4.0f) + 1.0f; break;
        case outInQuart:  y =  (x<0.5f) ?-8.0f*pow(x-0.5f,4.0f) + 0.5f : 
                                          8.0f*pow(x-0.5f,4.0f) + 0.5f; break;
   
        case inQuint:     y =                  pow(x     ,5.0f);        break;
        case outQuint:    y =                  pow(x-1.0f,5.0f) + 1.0f; break;
        case inOutQuint:  y =  (x<0.5f) ?16.0f*pow(x     ,5.0f) : 
                                         16.0f*pow(x-1.0f,5.0f) + 1.0f; break;
        case outInQuint:  y =            16.0f*pow(x-0.5f,5.0f) + 0.5f; break;

        case inSine:      y =      sin(x*SL_PI*0.5f- SL_PI*0.5f)+ 1.0f; break;
        case outSine:     y =      sin(x*SL_PI*0.5f);                   break;
        case inOutSine:   y = 0.5f*sin(x*SL_PI - SL_PI*0.5f) + 0.5f;    break;
        case outInSine:   y = (x<0.5f) ? 
                            0.5f*sin(x*SL_PI) : 
                            0.5f*sin(x*SL_PI - SL_PI) + 1.0f;         break;                                  
        default: y = x; 
    }

    SLfloat newTime = y * _totalTime;
    //printf("time: %5.3f, easing: %5.3f\n", time, newTime);

    return newTime;
}
//-----------------------------------------------------------------------------
