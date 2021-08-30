//#############################################################################
//  File:      SLCone.cpp
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLCone.h>

//-----------------------------------------------------------------------------
/*!
SLCone::SLCone ctor for conic revolution object around the z-axis
*/
SLCone::SLCone(SLAssetManager* assetMgr,
               SLfloat         coneRadius,
               SLfloat         coneHeight,
               SLuint          stacks,
               SLuint          slices,
               SLbool          hasBottom,
               SLstring        name,
               SLMaterial*     mat) : SLRevolver(assetMgr, name)
{
    assert(slices >= 3 && "Error: Not enough slices.");
    assert(slices > 0 && "Error: Not enough stacks.");

    _radius    = coneRadius;
    _height    = coneHeight;
    _stacks    = stacks;
    _hasBottom = hasBottom;

    _slices      = slices;
    _smoothFirst = hasBottom;
    _smoothLast  = false;
    _revAxis.set(0, 0, 1);
    SLuint nPoints = stacks + 1;
    if (hasBottom) nPoints += 2;
    _revPoints.reserve(nPoints);

    SLfloat dHeight = coneHeight / stacks;
    SLfloat h       = 0;
    SLfloat dRadius = coneRadius / stacks;
    SLfloat r       = coneRadius;

    if (hasBottom)
    { // define double points for sharp edges
        _revPoints.push_back(SLVec3f(0, 0, 0));
        _revPoints.push_back(SLVec3f(coneRadius, 0, 0));
    }
    for (SLuint i = 0; i <= stacks; ++i)
    {
        _revPoints.push_back(SLVec3f(r, 0, h));
        h += dHeight;
        r -= dRadius;
    }

    buildMesh(mat);
}
//-----------------------------------------------------------------------------
