//#############################################################################
//  File:      SLCurveBezier.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCURVEBEZIER_H
#define SLCURVEBEZIER_H

#include <SLCurve.h>
#include <SLGLVertexArrayExt.h>
#include <SLMat4.h>

//-----------------------------------------------------------------------------
//! The SLCurveBezier class implements a Bezier curve interpolation
/*!The SLCurveBezier class implements a Bezier curve interpolation. The math
is originally based on the implementation from the book:
"Essential Mathematics for Games and Interactive Applications" from
James M. Van Verth and Lars M. Bishop.
*/
class SLCurveBezier : public SLCurve
{
public:
    SLCurveBezier(const SLVVec4f& points);
    SLCurveBezier(const SLVVec4f& points,
                  const SLVVec3f& controlPoints);
    ~SLCurveBezier();

    void    dispose();
    void    init(const SLVVec4f& points,
                 const SLVVec3f& controlPoints);
    void    draw(const SLMat4f& wm);
    SLVec3f evaluate(const SLfloat t);
    SLVec3f velocity(SLfloat t);
    SLVec3f acceleration(SLfloat t);

    SLfloat segmentArcLength(SLuint  i,
                             SLfloat u1,
                             SLfloat u2);
    SLfloat subdivideLength(const SLVec3f& P0, const SLVec3f& P1, const SLVec3f& P2, const SLVec3f& P3);
    void    subdivideRender(SLVVec3f&      points,
                            const SLMat4f& wm,
                            SLfloat        epsilon,
                            const SLVec3f& P0,
                            const SLVec3f& P1,
                            const SLVec3f& P2,
                            const SLVec3f& P3);
    SLfloat arcLength(SLfloat t1, SLfloat t2);
    SLfloat findParamByDist(SLfloat t1, SLfloat s);

    // Getters
    SLint     numControlPoints() { return 2 * ((SLint)_points.size() - 1); }
    SLVVec3f& controls() { return _controls; }

protected:
    SLVVec3f           _controls; //!< Control points of Bezier curve
    SLGLVertexArrayExt _vao;      //!< Vertex array object for rendering
};
//-----------------------------------------------------------------------------
#endif
