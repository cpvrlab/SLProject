//#############################################################################
//  File:      SLPoints.cpp
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

#include <SLPoints.h>

//-----------------------------------------------------------------------------
//! SLPoints ctor with a givven vector of points
SLPoints::SLPoints(SLVVec3f& points,
                   SLstring name,
                   SLMaterial* material) : SLMesh(name)
{
    assert(name!="");

    _primitive = PT_points;

    if (points.size() > UINT_MAX)
        SL_EXIT_MSG("SLPoints supports max. 2^32 vertices.");

    P = points;

    mat = material;
}
//-----------------------------------------------------------------------------
//! SLPoints ctor for a random point cloud with the rnd generator.
SLPoints::SLPoints(SLfloat nPoints, SLRnd3f& rnd, SLstring name,
                   SLMaterial* material) : SLMesh(name)
{
    assert(name!="" && "No name provided in SLPoints!");
    assert(nPoints <= UINT_MAX && "SLPoints supports max. 2^32 vertices!");

    _primitive = PT_points;

    for (int i=0; i<nPoints; ++i)
        P.push_back(rnd.generate());

    mat = material;
}
//-----------------------------------------------------------------------------
