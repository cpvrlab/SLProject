//#############################################################################
//  File:      SLCylinder.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLCylinder.h>

//-----------------------------------------------------------------------------
/*!
SLCylinder::SLCylinder ctor for cylindric revolution object around the z-axis.
*/
SLCylinder::SLCylinder(SLfloat  cylinderRadius, 
                       SLfloat  cylinderHeight,
                       SLint    stacks, 
                       SLint    slices,
                       SLbool   hasTop, 
                       SLbool   hasBottom,
                       SLstring name,
                       SLMaterial*   mat) : SLRevolver(name)
{  
    assert(slices >= 3 && "Error: Not enough slices.");
    assert(slices >  0 && "Error: Not enough stacks.");
   
    _radius      = cylinderRadius;
    _height      = cylinderHeight;
    _stacks      = stacks;
    _hasTop      = hasTop;
    _hasBottom   = hasBottom;

    _slices      = slices;
    _smoothFirst = hasBottom;
    _smoothLast  = hasTop;
    _revAxis.set(0,0,1);
    SLint nPoints = stacks + 1;
    if (hasTop)    nPoints += 2;
    if (hasBottom) nPoints += 2;
    _revPoints.reserve(nPoints);
   
    SLfloat dHeight = cylinderHeight / stacks;
    SLfloat h = 0;
   
    if (hasBottom) 
    {   // define double points for sharp edges
        _revPoints.push_back(SLVec3f(0,0,0));
        _revPoints.push_back(SLVec3f(cylinderRadius, 0, 0));
    }

    for (SLint i=0; i<=stacks; ++i)
    {   _revPoints.push_back(SLVec3f(cylinderRadius, 0, h));
        h += dHeight;
    }

    if (hasTop) 
    {   // define double points for sharp edges
        _revPoints.push_back(SLVec3f(cylinderRadius, 0, cylinderHeight));  
        _revPoints.push_back(SLVec3f(0,              0, cylinderHeight));
    }
   
    buildMesh(mat);
}
//-----------------------------------------------------------------------------
