//#############################################################################
//  File:      math/SLCurveBezier.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLCurveBezier.h>
#include <SLScene.h>

//-------------------------------------------------------------------------------
SLCurveBezier::SLCurveBezier(const SLVec3f*  points,
                             const SLfloat*  times,
                             const SLint     numPointsAndTimes,
                             const SLVec3f*  controlPoints)
{
    _points = 0;
    _controls = 0;
    _times = 0;
    _lengths = 0;
    _totalLength = 0.0f;
    _count = 0;

    init(points, times, numPointsAndTimes, controlPoints);
}
//-------------------------------------------------------------------------------
SLCurveBezier::~SLCurveBezier()
{
   dispose();
}
//-------------------------------------------------------------------------------
/*! 
Init curve with curve points and times (pointsAndTimes.w = time). If no control
points are passed they will be calculated automatically
@param points Array of points on the Beziér curve
@param times Array times for the according Bézier point
@param numPointsAndTimes NO. of points in the arrays
@param controlPoints Array of control points with size = 2*(numPointsAndTimes-1)
*/
void SLCurveBezier::init(const SLVec3f*  points,
                         const SLfloat*  times,
                         const SLint     numPointsAndTimes,
                         const SLVec3f*  controlPoints)
{
    assert(numPointsAndTimes > 1);

    dispose();
   
    // set up arrays
    _count     = numPointsAndTimes;
    _points    = new SLVec3f[_count];
    _times     = new SLfloat[_count];
    _controls  = new SLVec3f[2*(_count-1)];

    // copy interpolant data
    unsigned int i;
    for (i = 0; i < _count; ++i)
    {   _points[i] = points[i];
        _times[i]  = times[i];
    }

    if (controlPoints == 0)
    {  
        if (_count > 2)
        {   // create approximating control points
            for (i = 0; i < _count-1; ++i)
            {   if (i > 0)
                    _controls[2*i  ] = _points[i  ] + (_points[i+1]-_points[i-1])/3.0f;
                if (i < _count-2)
                    _controls[2*i+1] = _points[i+1] - (_points[i+2]-_points[i  ])/3.0f;
            }

            _controls[0] = _controls[1] - (_points[1] - _points[0])/3.0f;
            _controls[2*_count-3] = _controls[2*_count-4] + 
                                    (_points[_count-1] - _points[_count-2])/3.0f;
        } else
        {   _controls[0] = _points[0] + (_points[1]-_points[0])/3.0f;
            _controls[1] = _points[1] - (_points[1]-_points[0])/3.0f;
        }
    }
    else
    {  // copy approximating control points
        for (i = 0; i < 2*(_count-1); ++i)
            _controls[i] = controlPoints[i];
    }

    // set up curve segment lengths
    _lengths = new SLfloat[_count-1];
    _totalLength = 0.0f;

    _totalLength = 0.0f;
    for (SLuint i = 0; i < _count-1; ++i)
    {   _lengths[i] = segmentArcLength(i, 0.0f, 1.0f);
        _totalLength += _lengths[i];
    }
}
//-----------------------------------------------------------------------------
/*! 
SLCurveBezier::draw does the OpenGL rendering of the Bezier curve in world 
space.
*/
void SLCurveBezier::draw(SLMat4f &wm)
{  
    SLint numControlPoints = 2*(_count-1);

    // Create buffer object
    if (!_bufP.id())
    {  
        // Build renderPoints by recursively subdividing the curve
        SLVVec3f renderPoints;
        for (SLuint i = 0; i < _count-1; ++i)
        {  subdivideRender(renderPoints, wm, 0.00001f, 
                            _points[i], _controls[2*i], 
                            _controls[2*i+1], _points[i+1]);
        }
   
        // add last point to the curve vector
        renderPoints.push_back(wm.multVec(_points[_count-1]));

        // add inputs points
        for (SLuint i = 0; i < _count; ++i)
            renderPoints.push_back(wm.multVec(_points[i]));
      
        // add control points
        for (SLint i = 0; i < numControlPoints; ++i)
            renderPoints.push_back(wm.multVec(_controls[i]));

        // add tangent points
        for (SLint i = 0; i < numControlPoints; ++i)
        {   renderPoints.push_back(wm.multVec(_controls[i]));
            int iPoint = (SLint)((SLfloat)i/2.0f + 0.5f);
            renderPoints.push_back(wm.multVec(_points[iPoint]));
        }
      
        // Generate finally the OpenGL rendering buffer
        _bufP.generate(&renderPoints[0], (SLint)renderPoints.size(), 3);
    }
   
    if (!_bufP.id()) return;

    SLint numTangentPoints = numControlPoints * 2;
    SLint numCurvePoints = _bufP.numElements() -
                           _count - numControlPoints - numTangentPoints;
   
    // Draw curve as a line strip through interpolated points
    _bufP.drawArrayAsConstantColorLineStrip(SLCol3f::RED, 1, 0, numCurvePoints);
   
    // ES2 has often problems with rendering points
    #ifndef SL_GLES2
    // Draw curve as a line strip through interpolated points
    _bufP.drawArrayAsConstantColorPoints(SLCol3f::RED, 3, 0, numCurvePoints);

    // Draw input points
    _bufP.drawArrayAsConstantColorPoints(SLCol3f::BLUE, 6, numCurvePoints, _count);

    // Draw control points
    _bufP.drawArrayAsConstantColorPoints(SLCol3f::YELLOW, 6,
        numCurvePoints + _count, numControlPoints);

    // Draw tangent points as lines
    _bufP.drawArrayAsConstantColorLines(SLCol3f::YELLOW, 1,
        numCurvePoints + _count + numControlPoints, numTangentPoints);
    #endif
}
//-------------------------------------------------------------------------------
//! Deletes all curve arrays
void SLCurveBezier::dispose()
{
    delete [] _points;   
    delete [] _controls; 
    delete [] _times;    
    delete [] _lengths;

    _points = 0;
    _controls = 0;
    _times = 0;
    _lengths = 0;

    _totalLength = 0.0f;
    _count = 0;
}
//-------------------------------------------------------------------------------
/*!
SLCurveBezier::curveEvaluate determines the position on the curve at time t.
*/
SLVec3f SLCurveBezier::evaluate(const SLfloat t)
{
    assert(_count > 1 && _points && _times);

    // handle boundary conditions
    if (t <= _times[0]) return _points[0]; else 
    if (t >= _times[_count-1]) return _points[_count-1];

    // find segment and parameter
    unsigned int i;
    for (i = 0; i < _count-1; ++i)
        if (t < _times[i+1]) 
            break;

    SLfloat t0 = _times[i];
    SLfloat t1 = _times[i+1];
    SLfloat u  = (t - t0)/(t1 - t0);

    // evaluate
    SLVec3f A =        _points[i+1]
                - 3.0f*_controls[2*i+1]
                + 3.0f*_controls[2*i]
                -      _points[i];
    SLVec3f B =   3.0f*_controls[2*i+1]
                - 6.0f*_controls[2*i]
                + 3.0f*_points[i];
    SLVec3f C =   3.0f*_controls[2*i]
                - 3.0f*_points[i];
    
    return _points[i] + u*(C + u*(B + u*A));
}
//-------------------------------------------------------------------------------
/*!
SLCurveBezier::curveVelocity determines the velocity vector on the curve at 
point t. The velocity vector direction is the tangent vector at t an is the first 
derivative at point t. The velocity vector magnitute is the speed at point t. 
*/
SLVec3f SLCurveBezier::velocity(SLfloat t)
{
    assert(_count > 1 && _points && _times);

    // handle boundary conditions
    if (t <= _times[0])
        return _points[0];
    else if (t >= _times[_count-1])
        return _points[_count-1];

    // find segment and parameter
    unsigned int i;
    for (i = 0; i < _count-1; ++i)
        if (t < _times[i+1])
            break;

    SLfloat t0 = _times[i];
    SLfloat t1 = _times[i+1];
    SLfloat u  = (t - t0)/(t1 - t0);

    // evaluate
    SLVec3f A = _points[i+1]
                - 3.0f*_controls[2*i+1]
                + 3.0f*_controls[2*i]
                - _points[i];
    SLVec3f B = 6.0f*_controls[2*i+1]
                - 12.0f*_controls[2*i]
                + 6.0f*_points[i];
    SLVec3f C = 3.0f*_controls[2*i]
                - 3.0f*_points[i];
    
    return C + u*(B + 3.0f*u*A);

}
//-------------------------------------------------------------------------------
/*!
SLCurveBezier::curveAcceleration determines the acceleration vector on the curve
at time t. It is the second derivative at point t.
*/
SLVec3f SLCurveBezier::acceleration(SLfloat t)
{
    assert(_count > 1 && _points && _times);

    // handle boundary conditions
    if (t <= _times[0])
        return _points[0];
    else if (t >= _times[_count-1])
        return _points[_count-1];

    // find segment and parameter
    unsigned int i;
    for (i = 0; i < _count-1; ++i)
        if (t < _times[i+1]) break;

    SLfloat t0 = _times[i];
    SLfloat t1 = _times[i+1];
    SLfloat u  = (t - t0)/(t1 - t0);

    // evaluate
    SLVec3f A =         _points[i+1]
                -  3.0f*_controls[2*i+1]
                +  3.0f*_controls[2*i]
                -       _points[i];
    SLVec3f B =    6.0f*_controls[2*i+1]
                - 12.0f*_controls[2*i]
                +  6.0f*_points[i];
    
    return B + 6.0f*u*A;

}
//-------------------------------------------------------------------------------
/*!
SLCurveBezier::findParamByDist gets parameter s distance in arc length from Q(t1).
Returns max SLfloat if can't find it.
*/
SLfloat SLCurveBezier::findParamByDist(SLfloat t1, SLfloat s)
{
    // ensure that we remain within valid parameter space
    if (s > arcLength(t1, _times[_count-1]))
        return _times[_count-1];

    // make first guess
    SLfloat p = t1 + s*(_times[_count-1]-_times[0])/_totalLength;
    for (SLuint i = 0; i < 32; ++i)
    {
        // compute function value and test against zero
        SLfloat func = arcLength(t1, p) - s;
        if (SL_abs(func) < 1.0e-03f) return p;

        // perform Newton-Raphson iteration step
        SLfloat speed = velocity(p).length();
        assert(SL_abs(speed) > FLT_EPSILON);
        p -= func/speed;
    }

    // done iterating, return failure case
    return FLT_MAX;
}
//-------------------------------------------------------------------------------
/*! 
Calculate length of curve between parameters t1 and t2
*/
SLfloat SLCurveBezier::arcLength(SLfloat t1, SLfloat t2)
{
    if (t2 <= t1) return 0.0f;
    if (t1 < _times[0]) t1 = _times[0];
    if (t2 > _times[_count-1]) t2 = _times[_count-1];

    // find segment and parameter
    unsigned int seg1;
    for (seg1 = 0; seg1 < _count-1; ++seg1)
        if (t1 < _times[seg1+1])
            break;

    SLfloat u1 = (t1 - _times[seg1])/(_times[seg1+1] - _times[seg1]);
    
    // find segment and parameter
    unsigned int seg2;
    for (seg2 = 0; seg2 < _count-1; ++seg2)
    if (t2 <= _times[seg2+1])
        break;

    SLfloat u2 = (t2 - _times[seg2])/(_times[seg2+1] - _times[seg2]);
    
    // both parameters lie in one segment
    SLfloat result;
    if (seg1 == seg2)
    result = segmentArcLength(seg1, u1, u2);

    // parameters cross segments
    else
    {   result = segmentArcLength(seg1, u1, 1.0f);
        for (SLuint i = seg1+1; i < seg2; ++i)
            result += _lengths[i];
        result += segmentArcLength(seg2, 0.0f, u2);
    }

    return result;
}
//-------------------------------------------------------------------------------
/*!
SLCurveBezier::segmentArcLength calculate length of curve segment between 
parameters u1 and u2 by recursively subdividing the segment.
*/
SLfloat SLCurveBezier::segmentArcLength(SLuint  i, SLfloat u1, SLfloat u2)
{
    assert(i >= 0 && i < _count-1);

    if (u2 <= u1) return 0.0f;
    if (u1 < 0.0f) u1 = 0.0f;
    if (u2 > 1.0f) u2 = 1.0f;

    SLVec3f P0 = _points[i];
    SLVec3f P1 = _controls[2*i];
    SLVec3f P2 = _controls[2*i+1];
    SLVec3f P3 = _points[i+1];

    // get control points for subcurve from 0.0 to u2 (de Casteljau's method)
    // http://de.wikipedia.org/wiki/De-Casteljau-Algorithmus
    SLfloat minus_u2 = (1.0f - u2);
    SLVec3f L1 = minus_u2*P0 + u2*P1;
    SLVec3f H  = minus_u2*P1 + u2*P2;
    SLVec3f L2 = minus_u2*L1 + u2*H;
    SLVec3f L3 = minus_u2*L2 + u2*(minus_u2*H + u2*(minus_u2*P2 + u2*P3));

    // resubdivide to get control points for subcurve between u1 and u2
    SLfloat minus_u1 = (1.0f - u1);
    H = minus_u1*L1 + u1*L2;
    SLVec3f R3 = L3;
    SLVec3f R2 = minus_u1*L2 + u1*L3;
    SLVec3f R1 = minus_u1*H + u1*R2;
    SLVec3f R0 = minus_u1*(minus_u1*(minus_u1*P0 + u1*L1) + u1*H) + u1*R1;

    // get length through subdivision
    return subdivideLength(R0, R1, R2, R3);
}
//-------------------------------------------------------------------------------
/*!
SLCurveBezier::subdivideLength calculates length of Bezier curve by recursive
midpoint subdivision.
*/
SLfloat SLCurveBezier::subdivideLength(const SLVec3f& P0, 
                                       const SLVec3f& P1, 
                                       const SLVec3f& P2, 
                                       const SLVec3f& P3)
{
    // check to see if basically straight
    SLfloat Lmin = P0.distance(P3);
    SLfloat Lmax = P0.distance(P1) + P1.distance(P2) + P2.distance(P3);
    SLfloat diff = Lmin - Lmax;

    if (diff*diff < 1.0e-3f)
        return 0.5f*(Lmin + Lmax);

    // otherwise get control points for subdivision
    SLVec3f L1  = (P0 + P1) * 0.5f;
    SLVec3f H   = (P1 + P2) * 0.5f;
    SLVec3f L2  = (L1 + H ) * 0.5f;
    SLVec3f R2  = (P2 + P3) * 0.5f;
    SLVec3f R1  = (H  + R2) * 0.5f;
    SLVec3f mid = (L2 + R1) * 0.5f;

    // subdivide
    return subdivideLength(P0, L1, L2, mid) + subdivideLength(mid, R1, R2, P3);
}
//-------------------------------------------------------------------------------
/*!
SLCurveBezier::subdivideRender adds points along the curve to the point vector
renderPoints by recursively subdividing the curve with the Casteljau scheme.
*/
void SLCurveBezier::subdivideRender(SLVVec3f &renderPoints,
                                    SLMat4f &wm,
                                    SLfloat epsilon,
                                    SLVec3f& P0, SLVec3f& P1, 
                                    SLVec3f& P2, SLVec3f& P3)
{
    // add first point transformed by wm if not allready in the list
    if (renderPoints.size()==0)
        renderPoints.push_back(wm.multVec(P0));
    else if (P0 != renderPoints[renderPoints.size()-1])
        renderPoints.push_back(wm.multVec(P0));
   
    // check to see if basically straight
    SLfloat Lmin = P0.distance(P3);
    SLfloat Lmax = P0.distance(P1) + P1.distance(P2) + P2.distance(P3);
    SLfloat diff = Lmin - Lmax;
    if (diff*diff < epsilon) return;

    // otherwise get control points for subdivision
    SLVec3f L1  = (P0 + P1) * 0.5f;
    SLVec3f H   = (P1 + P2) * 0.5f;
    SLVec3f L2  = (L1 + H)  * 0.5f;
    SLVec3f R2  = (P2 + P3) * 0.5f;
    SLVec3f R1  = (H  + R2) * 0.5f;
    SLVec3f mid = (L2 + R1) * 0.5f;

    // subdivide
    subdivideRender(renderPoints, wm, epsilon, P0, L1, L2, mid);
    subdivideRender(renderPoints, wm, epsilon, mid, R1, R2, P3);
}
//-------------------------------------------------------------------------------
