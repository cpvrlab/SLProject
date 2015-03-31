//#############################################################################
//  File:      SLSphere.cpp
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

#include "SLSphere.h"

//-----------------------------------------------------------------------------
/*!
SLSphere::SLSphere ctor for spheric revolution object around the z-axis.
*/
SLSphere::SLSphere(SLfloat sphereRadius,
                     SLint stacks, 
                     SLint slices,
                     SLstring name,
                     SLMaterial* mat) : SLRevolver(name)
{
    assert(slices >= 3 && "Error: Not enough slices.");
    assert(slices >  0 && "Error: Not enough stacks.");
   
    _radius = sphereRadius;
    _stacks = stacks;

    _slices = slices;
    _smoothFirst = true;
    _smoothLast  = true;
    _revAxis.set(0,0,1);
    _revPoints.reserve(stacks+1);

    SLfloat theta = -SL_PI;
    SLfloat phi   = 0;
    SLfloat dTheta= SL_PI / stacks;
   
    for (SLint i=0; i<=stacks; ++i)
    {   SLVec3f p;
        p.fromSpherical(sphereRadius, theta, phi);
        _revPoints.push_back(p);
        theta += dTheta;
    }
   
    buildMesh(mat);
}
//-----------------------------------------------------------------------------
