//#############################################################################
//  File:      math/SLCurveBezier.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCURVEBEZIER_H
#define SLCURVEBEZIER_H

#include <SLCurve.h>
#include <SLMat4.h>
#include <SLGLBuffer.h>

//-----------------------------------------------------------------------------
//!The SLCurveBezier class implements a Bezier curve interpolation
/*!The SLCurveBezier class implements a Bezier curve interpolation. The math
is originally based on the implementation from the book:
"Essential Mathematics for Games and Interactive Applications" from 
James M. Van Verth and Lars M. Bishop.
*/
class SLCurveBezier : public SLCurve
{
    public:
                    SLCurveBezier     (const SLVec3f*  points,
                                        const SLfloat*  times,
                                        const SLint     numPointsAndTimes,
                                        const SLVec3f*  controlPoints=0);
                    ~SLCurveBezier     ();

        void        dispose           ();
        void        init              (const SLVec3f*  points,
                                        const SLfloat*  times,
                                        const SLint     numPointsAndTimes,
                                        const SLVec3f*  controlPoints);
        void        draw              (SLMat4f &wm);
        SLVec3f     evaluate          (const SLfloat t);
        SLVec3f     velocity          (SLfloat t);
        SLVec3f     acceleration      (SLfloat t);

        SLfloat     segmentArcLength  (SLuint i, 
                                        SLfloat u1, SLfloat u2);
        SLfloat     subdivideLength   (const SLVec3f& P0, const SLVec3f& P1, 
                                        const SLVec3f& P2, const SLVec3f& P3);
        void        subdivideRender   (SLVVec3f &points,
                                        SLMat4f &wm,
                                        SLfloat epsilon,
                                        SLVec3f& P0, SLVec3f& P1, 
                                        SLVec3f& P2, SLVec3f& P3);
        SLfloat     arcLength         (SLfloat t1, SLfloat t2);
        SLfloat     findParamByDist   (SLfloat t1, SLfloat s);

        // Getters
        SLint       numControlPoints  (){return 2*(_count-1);}
        SLVec3f*    controls          (){return _controls;}

    protected:
        SLVec3f*    _controls;        //!< Control points of Bézier curve
        SLGLBuffer  _bufP;            //!< Buffer for vertex positions
};
//-----------------------------------------------------------------------------
#endif
