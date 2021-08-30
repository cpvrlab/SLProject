//#############################################################################
//  File:      SLCylinder.cpp
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLCylinder.h>

#include <utility>

//-----------------------------------------------------------------------------
/*!
SLCylinder::SLCylinder ctor for cylindric revolution object around the z-axis.
*/
SLCylinder::SLCylinder(SLAssetManager* assetMgr,
                       SLfloat         cylinderRadius,
                       SLfloat         cylinderHeight,
                       SLuint          stacks,
                       SLuint          slices,
                       SLbool          hasTop,
                       SLbool          hasBottom,
                       SLstring        name,
                       SLMaterial*     mat) : SLRevolver(assetMgr, std::move(name))
{
    assert(slices >= 3 && "Error: Not enough slices.");
    assert(slices > 0 && "Error: Not enough stacks.");

    _radius    = cylinderRadius;
    _height    = cylinderHeight;
    _hasTop    = hasTop;
    _hasBottom = hasBottom;

    _stacks      = stacks;
    _slices      = slices;
    _smoothFirst = hasBottom;
    _smoothLast  = hasTop;
    _revAxis.set(0, 0, 1);
    SLuint nPoints = stacks + 1;
    if (hasTop) nPoints += 2;
    if (hasBottom) nPoints += 2;
    _revPoints.reserve(nPoints);

    SLfloat dHeight = cylinderHeight / stacks;
    SLfloat h       = 0;

    if (hasBottom)
    { // define double points for sharp edges
        _revPoints.push_back(SLVec3f(0, 0, 0));
        _revPoints.push_back(SLVec3f(cylinderRadius, 0, 0));
    }

    for (SLuint i = 0; i <= stacks; ++i)
    {
        _revPoints.push_back(SLVec3f(cylinderRadius, 0, h));
        h += dHeight;
    }

    if (hasTop)
    { // define double points for sharp edges
        _revPoints.push_back(SLVec3f(cylinderRadius, 0, cylinderHeight));
        _revPoints.push_back(SLVec3f(0, 0, cylinderHeight));
    }

    buildMesh(mat);
}
//-----------------------------------------------------------------------------
