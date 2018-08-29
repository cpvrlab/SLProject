//#############################################################################
//  File:      SLDisk.cpp
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

#include <SLDisk.h>

//-----------------------------------------------------------------------------
/*!
SLDisk::SLDisk ctor for disk revolution object around the z-axis
*/
SLDisk::SLDisk(SLfloat      radius,
               SLVec3f      revolveAxis,
               SLuint       slices,
               SLbool       doubleSided,
               SLstring     name,
               SLMaterial*  mat) : SLRevolver(name)
{  
    assert(slices >= 3 && "Error: Not enough slices.");
   
    _radius      = radius;
    _doubleSided = doubleSided;
    _slices      = slices;
    _smoothFirst = true;
    _revAxis     = revolveAxis;
   
    // add the centre & radius point
    if (_doubleSided)
    {   _revPoints.reserve(4);
        _revPoints.push_back(SLVec3f(0,0,0));
        _revPoints.push_back(SLVec3f(radius, 0, 0));
        _revPoints.push_back(SLVec3f(radius, 0, 0));
        _revPoints.push_back(SLVec3f(0,0,0));
        _smoothLast = true;
    } else
    {   _revPoints.reserve(2);
        _revPoints.push_back(SLVec3f(0,0,0));
        _revPoints.push_back(SLVec3f(radius, 0, 0));
        _smoothLast = false;
    }

    buildMesh(mat);
}
//-----------------------------------------------------------------------------
