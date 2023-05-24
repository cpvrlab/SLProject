//#############################################################################
//  File:      math/SLCurve.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCURVE_H
#define SLCURVE_H

#include <SLMath.h>
#include <SLMat4.h>
#include <SLVec3.h>
#include <SLVec4.h>

//-----------------------------------------------------------------------------
//!Base class for curves defined by multiple 3D points.
/*!Base class for curves defined by multiple 3D points. Derived classes (e.g.
SLCurveBezier) implement specific curve interpolation schemes.
*/
class SLCurve
{
public:
    SLCurve() {}
    virtual ~SLCurve() {}

    virtual void    dispose()                 = 0;
    virtual SLVec3f evaluate(const SLfloat t) = 0;
    virtual void    draw(const SLMat4f& wm)   = 0;

protected:
    SLVVec4f _points;      //!< Sample points (x,y,z) and time (w) of curve
    SLVfloat _lengths;     //!< Length of each curve segment
    SLfloat  _totalLength; //!< Total length of curve
};
//-----------------------------------------------------------------------------
#endif
