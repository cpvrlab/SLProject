//#############################################################################
//  File:      math/SLCurveBezier.cpp
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <math/SLCurveBezier.h>
#include <SLGLState.h>
#include <SLScene.h>

//-------------------------------------------------------------------------------
SLCurveBezier::SLCurveBezier(const SLVVec4f& points)
{
    _totalLength = 0.0f;
    SLVVec3f ctrls;
    init(points, ctrls);
}
//-------------------------------------------------------------------------------
SLCurveBezier::SLCurveBezier(const SLVVec4f& points,
                             const SLVVec3f& controlPoints)
{
    _totalLength = 0.0f;

    init(points, controlPoints);
}
//-------------------------------------------------------------------------------
SLCurveBezier::~SLCurveBezier()
{
    dispose();
}
//-------------------------------------------------------------------------------
/*!
Init curve with curve points and times. If no control
points are passed they will be calculated automatically
@param points Array of points on the Bezier curve (w = time)
@param controlPoints Array of control points with size = 2*(numPointsAndTimes-1)
*/
void SLCurveBezier::init(const SLVVec4f& points,
                         const SLVVec3f& controlPoints)
{
    assert(points.size() > 1);

    dispose();

    // set up arrays
    _points.clear();
    _points.resize(points.size());
    _controls.resize(2 * (points.size() - 1));

    // copy interpolated data
    SLuint i;
    for (i = 0; i < _points.size(); ++i)
    {
        _points[i] = points[i];
    }

    if (controlPoints.empty())
    {
        if (points.size() > 2)
        {
            // create approximating control points
            for (i = 0; i < _points.size() - 1; ++i)
            {
                if (i > 0)
                    _controls[2 * i] = (_points[i] + (_points[i + 1] - _points[i - 1]) / 3.0f).vec3();
                if (i < _points.size() - 2)
                    _controls[2 * i + 1] = (_points[i + 1] - (_points[i + 2] - _points[i]) / 3.0f).vec3();
            }

            _controls[0]                      = (SLVec4f(_controls[1]) - (_points[1] - _points[0]) / 3.0f).vec3();
            _controls[2 * _points.size() - 3] = (SLVec4f(_controls[2 * _points.size() - 4]) +
                                                 (_points[_points.size() - 1] - _points[_points.size() - 2]) / 3.0f)
                                                  .vec3();
        }
        else
        {
            _controls[0] = (_points[0] + (_points[1] - _points[0]) / 3.0f).vec3();
            _controls[1] = (_points[1] - (_points[1] - _points[0]) / 3.0f).vec3();
        }
    }
    else
    { // copy approximating control points
        for (i = 0; i < 2 * (_points.size() - 1); ++i)
            _controls[i] = controlPoints[i];
    }

    // set up curve segment lengths
    _lengths.clear();
    _lengths.resize(_points.size() - 1);
    _totalLength = 0.0f;

    _totalLength = 0.0f;
    for (SLuint i = 0; i < _points.size() - 1; ++i)
    {
        _lengths[i] = segmentArcLength(i, 0.0f, 1.0f);
        _totalLength += _lengths[i];
    }
}
//-----------------------------------------------------------------------------
/*!
SLCurveBezier::draw does the rendering of the Bezier curve in world space.
*/
void SLCurveBezier::draw(const SLMat4f& wm)
{
    SLint numControlPoints = 2 * ((SLint)_points.size() - 1);

    // Create buffer object
    if (!_vao.vaoID())
    {
        // Build renderPoints by recursively subdividing the curve
        SLVVec3f renderPoints;
        for (SLuint i = 0; i < _points.size() - 1; ++i)
        {
            subdivideRender(renderPoints,
                            wm,
                            0.00001f,
                            _points[i].vec3(),
                            _controls[2 * i],
                            _controls[2 * i + 1],
                            _points[i + 1].vec3());
        }

        // add last point to the curve vector
        renderPoints.push_back(wm.multVec(_points.back().vec3()));

        // add inputs points
        for (SLuint i = 0; i < _points.size(); ++i)
            renderPoints.push_back(wm.multVec(_points[i].vec3()));

        // add control points
        for (SLuint i = 0; i < (SLuint)numControlPoints; ++i)
            renderPoints.push_back(wm.multVec(_controls[i]));

        // add tangent points
        for (SLuint i = 0; i < (SLuint)numControlPoints; ++i)
        {
            renderPoints.push_back(wm.multVec(_controls[i]));
            int iPoint = (SLint)((SLfloat)i / 2.0f + 0.5f);
            renderPoints.push_back(wm.multVec(_points[(SLuint)iPoint].vec3()));
        }

        // Generate finally the OpenGL rendering buffer
        _vao.generateVertexPos(&renderPoints);
    }

    if (!_vao.vaoID()) return;

    // Set the view transform
    SLGLState* stateGL = SLGLState::instance();
    stateGL->modelMatrix.identity();

    SLint numTangentPoints = numControlPoints * 2;
    SLint numCurvePoints   = (SLint)_vao.numVertices() -
                           (SLint)_points.size() -
                           numControlPoints -
                           numTangentPoints;

    // Draw curve as a line strip through interpolated points
    _vao.drawArrayAsColored(PT_lineStrip,
                            SLCol4f::RED,
                            1,
                            0,
                            (SLuint)numCurvePoints);

// ES2 has often problems with rendering points
#ifndef APP_USES_GLES
    // Draw curve as a line strip through interpolated points
    _vao.drawArrayAsColored(PT_points,
                            SLCol4f::RED,
                            3,
                            0,
                            (SLuint)numCurvePoints);

    // Draw input points
    _vao.drawArrayAsColored(PT_points,
                            SLCol4f::BLUE,
                            6,
                            (SLuint)numCurvePoints,
                            (SLuint)_points.size());

    // Draw control points
    _vao.drawArrayAsColored(PT_points,
                            SLCol4f::YELLOW,
                            6,
                            (SLuint)numCurvePoints + (SLuint)_points.size(),
                            (SLuint)numControlPoints);

    // Draw tangent points as lines
    _vao.drawArrayAsColored(PT_lines,
                            SLCol4f::YELLOW,
                            1,
                            (SLuint)numCurvePoints +
                              (SLuint)_points.size() +
                              (SLuint)numControlPoints,
                            (SLuint)numTangentPoints);
#endif
}
//-------------------------------------------------------------------------------
//! Deletes all curve arrays
void SLCurveBezier::dispose()
{
    _points.clear();
    _lengths.clear();
    _totalLength = 0.0f;
}
//-------------------------------------------------------------------------------
/*!
SLCurveBezier::curveEvaluate determines the position on the curve at time t.
*/
SLVec3f SLCurveBezier::evaluate(const SLfloat t)
{
    assert(_points.size() > 1);

    // handle boundary conditions
    if (t <= _points[0].w)
        return _points[0].vec3();
    else if (t >= _points.back().w)
        return _points.back().vec3();

    // find segment and parameter
    unsigned int i;
    for (i = 0; i < _points.size() - 1; ++i)
        if (t < _points[i + 1].w)
            break;

    SLfloat t0 = _points[i].w;
    SLfloat t1 = _points[i + 1].w;
    SLfloat u  = (t - t0) / (t1 - t0);

    // evaluate
    SLVec3f A = _points[i + 1].vec3() -
                3.0f * _controls[2 * i + 1] +
                3.0f * _controls[2 * i] -
                _points[i].vec3();
    SLVec3f B = 3.0f * _controls[2 * i + 1] -
                6.0f * _controls[2 * i] +
                3.0f * _points[i].vec3();
    SLVec3f C = 3.0f * _controls[2 * i] -
                3.0f * _points[i].vec3();

    return _points[i].vec3() + u * (C + u * (B + u * A));
}
//-------------------------------------------------------------------------------
/*!
SLCurveBezier::curveVelocity determines the velocity vector on the curve at
point t. The velocity vector direction is the tangent vector at t an is the first
derivative at point t. The velocity vector magnitude is the speed at point t.
*/
SLVec3f SLCurveBezier::velocity(SLfloat t)
{
    assert(_points.size() > 1);

    // handle boundary conditions
    if (t <= _points[0].w)
        return _points[0].vec3();
    else if (t >= _points.back().w)
        return _points.back().vec3();

    // find segment and parameter
    unsigned int i;
    for (i = 0; i < _points.size() - 1; ++i)
        if (t < _points[i + 1].w)
            break;

    SLfloat t0 = _points[i].w;
    SLfloat t1 = _points[i + 1].w;
    SLfloat u  = (t - t0) / (t1 - t0);

    // evaluate
    SLVec3f A = _points[i + 1].vec3() -
                3.0f * _controls[2 * i + 1] +
                3.0f * _controls[2 * i] -
                _points[i].vec3();
    SLVec3f B = 6.0f * _controls[2 * i + 1] -
                12.0f * _controls[2 * i] +
                6.0f * _points[i].vec3();
    SLVec3f C = 3.0f * _controls[2 * i] -
                3.0f * _points[i].vec3();

    return C + u * (B + 3.0f * u * A);
}
//-------------------------------------------------------------------------------
/*!
SLCurveBezier::curveAcceleration determines the acceleration vector on the curve
at time t. It is the second derivative at point t.
*/
SLVec3f SLCurveBezier::acceleration(SLfloat t)
{
    assert(_points.size() > 1);

    // handle boundary conditions
    if (t <= _points[0].w)
        return _points[0].vec3();
    else if (t >= _points.back().w)
        return _points.back().vec3();

    // find segment and parameter
    unsigned int i;
    for (i = 0; i < _points.size() - 1; ++i)
        if (t < _points[i + 1].w) break;

    SLfloat t0 = _points[i].w;
    SLfloat t1 = _points[i + 1].w;
    SLfloat u  = (t - t0) / (t1 - t0);

    // evaluate
    SLVec3f A = _points[i + 1].vec3() -
                3.0f * _controls[2 * i + 1] +
                3.0f * _controls[2 * i] -
                _points[i].vec3();
    SLVec3f B = 6.0f * _controls[2 * i + 1] -
                12.0f * _controls[2 * i] +
                6.0f * _points[i].vec3();

    return B + 6.0f * u * A;
}
//-------------------------------------------------------------------------------
/*!
SLCurveBezier::findParamByDist gets parameter s distance in arc length from Q(t1).
Returns max SLfloat if can't find it.
*/
SLfloat SLCurveBezier::findParamByDist(SLfloat t1, SLfloat s)
{
    // ensure that we remain within valid parameter space
    if (s > arcLength(t1, _points.back().w))
        return _points.back().w;

    // make first guess
    SLfloat p = t1 + s * (_points.back().w - _points[0].w) / _totalLength;
    for (SLuint i = 0; i < 32; ++i)
    {
        // compute function value and test against zero
        SLfloat func = arcLength(t1, p) - s;
        if (Utils::abs(func) < 1.0e-03f) return p;

        // perform Newton-Raphson iteration step
        SLfloat speed = velocity(p).length();
        assert(Utils::abs(speed) > FLT_EPSILON);
        p -= func / speed;
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
    if (t1 < _points[0].w) t1 = _points[0].w;
    if (t2 > _points.back().w) t2 = _points.back().w;

    // find segment and parameter
    unsigned int seg1;
    for (seg1 = 0; seg1 < _points.size() - 1; ++seg1)
        if (t1 < _points[seg1 + 1].w)
            break;

    SLfloat u1 = (t1 - _points[seg1].w) / (_points[seg1 + 1].w - _points[seg1].w);

    // find segment and parameter
    unsigned int seg2;
    for (seg2 = 0; seg2 < _points.size() - 1; ++seg2)
        if (t2 <= _points[seg2 + 1].w)
            break;

    SLfloat u2 = (t2 - _points[seg2].w) / (_points[seg2 + 1].w - _points[seg2].w);

    // both parameters lie in one segment
    SLfloat result;
    if (seg1 == seg2)
        result = segmentArcLength(seg1, u1, u2);

    // parameters cross segments
    else
    {
        result = segmentArcLength(seg1, u1, 1.0f);
        for (SLuint i = seg1 + 1; i < seg2; ++i)
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
SLfloat SLCurveBezier::segmentArcLength(SLuint i, SLfloat u1, SLfloat u2)
{
    assert(i >= 0 && i < _points.size() - 1);

    if (u2 <= u1) return 0.0f;
    if (u1 < 0.0f) u1 = 0.0f;
    if (u2 > 1.0f) u2 = 1.0f;

    SLVec3f P0 = _points[i].vec3();
    SLVec3f P1 = _controls[2 * i];
    SLVec3f P2 = _controls[2 * i + 1];
    SLVec3f P3 = _points[i + 1].vec3();

    // get control points for subcurve from 0.0 to u2 (de Casteljau's method)
    // http://de.wikipedia.org/wiki/De-Casteljau-Algorithmus
    SLfloat minus_u2 = (1.0f - u2);
    SLVec3f L1       = minus_u2 * P0 + u2 * P1;
    SLVec3f H        = minus_u2 * P1 + u2 * P2;
    SLVec3f L2       = minus_u2 * L1 + u2 * H;
    SLVec3f L3       = minus_u2 * L2 + u2 * (minus_u2 * H + u2 * (minus_u2 * P2 + u2 * P3));

    // resubdivide to get control points for subcurve between u1 and u2
    SLfloat minus_u1 = (1.0f - u1);
    H                = minus_u1 * L1 + u1 * L2;
    SLVec3f R3       = L3;
    SLVec3f R2       = minus_u1 * L2 + u1 * L3;
    SLVec3f R1       = minus_u1 * H + u1 * R2;
    SLVec3f R0       = minus_u1 * (minus_u1 * (minus_u1 * P0 + u1 * L1) + u1 * H) + u1 * R1;

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

    if (diff * diff < 1.0e-3f)
        return 0.5f * (Lmin + Lmax);

    // otherwise get control points for subdivision
    SLVec3f L1  = (P0 + P1) * 0.5f;
    SLVec3f H   = (P1 + P2) * 0.5f;
    SLVec3f L2  = (L1 + H) * 0.5f;
    SLVec3f R2  = (P2 + P3) * 0.5f;
    SLVec3f R1  = (H + R2) * 0.5f;
    SLVec3f mid = (L2 + R1) * 0.5f;

    // subdivide
    return subdivideLength(P0, L1, L2, mid) + subdivideLength(mid, R1, R2, P3);
}
//-------------------------------------------------------------------------------
/*!
SLCurveBezier::subdivideRender adds points along the curve to the point vector
renderPoints by recursively subdividing the curve with the Casteljau scheme.
*/
void SLCurveBezier::subdivideRender(SLVVec3f&      renderPoints,
                                    const SLMat4f& wm,
                                    SLfloat        epsilon,
                                    const SLVec3f& P0,
                                    const SLVec3f& P1,
                                    const SLVec3f& P2,
                                    const SLVec3f& P3)
{
    // add first point transformed by wm if not already in the list
    if (renderPoints.empty())
        renderPoints.push_back(wm.multVec(P0));
    else if (P0 != renderPoints.back())
        renderPoints.push_back(wm.multVec(P0));

    // check to see if basically straight
    SLfloat Lmin = P0.distance(P3);
    SLfloat Lmax = P0.distance(P1) + P1.distance(P2) + P2.distance(P3);
    SLfloat diff = Lmin - Lmax;
    if (diff * diff < epsilon) return;

    // otherwise get control points for subdivision
    SLVec3f L1  = (P0 + P1) * 0.5f;
    SLVec3f H   = (P1 + P2) * 0.5f;
    SLVec3f L2  = (L1 + H) * 0.5f;
    SLVec3f R2  = (P2 + P3) * 0.5f;
    SLVec3f R1  = (H + R2) * 0.5f;
    SLVec3f mid = (L2 + R1) * 0.5f;

    // subdivide
    subdivideRender(renderPoints, wm, epsilon, P0, L1, L2, mid);
    subdivideRender(renderPoints, wm, epsilon, mid, R1, R2, P3);
}
//-------------------------------------------------------------------------------
