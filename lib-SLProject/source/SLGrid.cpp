//#############################################################################
//  File:      SLGrid.cpp
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

#include <SLGrid.h>

//-----------------------------------------------------------------------------
//! SLGrid ctor with min & max corners and its resolutions 
SLGrid::SLGrid(SLVec3f minXZ, SLVec3f maxXZ,
               SLuint resX, SLuint resZ, 
               SLstring name,
               SLMaterial* mat) : SLMesh(name) 
{
    assert(minXZ!=maxXZ);
    assert(resX>0);
    assert(resZ>0);
    assert(name!="");
    assert(minXZ.y==0 && maxXZ.y==0);

    _primitive = SL_LINES;
    _min = minXZ;
    _max = maxXZ;
    _resX = resX;
    _resZ = resZ;
    _isVolume = false;

    buildMesh(mat);
}
//-----------------------------------------------------------------------------
//! SLGrid::buildMesh fills in the underlying arrays from the SLMesh object
void SLGrid::buildMesh(SLMaterial* material)
{  
    deleteData();
   
    // Check max. allowed no. of verts
    SLuint64 uIntNumV64 = (_resX+1) * 2 + (_resZ+1) * 2;
    if (uIntNumV64 > UINT_MAX)
        SL_EXIT_MSG("SLGrid supports max. 2^32 vertices.");

    // allocate new arrays of SLMesh
    numV = (_resX+1) * 2 + (_resZ+1) * 2;
    numI = numV;
    P = new SLVec3f[numV];

    if (uIntNumV64 < 65535)
         I16 = new SLushort[numI];
    else I32 = new SLuint[numI];

    //Set one default material index
    mat = material;
   
    // delta vector
    SLVec3f d = _max - _min;
    d.x /= (SLfloat)_resX;
    d.z /= (SLfloat)_resZ;
   
    // Build vertex data
    SLuint p = 0;
    for (SLuint x=0; x<=_resX; ++x)
    {   P[p++].set(_min.x + x*d.x, 0, _min.z);
        P[p++].set(_min.x + x*d.x, 0, _max.z);
    }
    for (SLuint z=0; z<=_resZ; ++z)
    {   P[p++].set(_min.x, 0, _min.z + z*d.z);
        P[p++].set(_max.x, 0, _min.z + z*d.z);
    }

    // Indexes
    if (I16)
         for (SLuint i=0; i < numI; ++i) I16[i] = i;
    else for (SLuint i=0; i < numI; ++i) I32[i] = i;
}
//-----------------------------------------------------------------------------
