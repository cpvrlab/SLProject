//#############################################################################
//  File:      SLGrid.cpp
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <limits.h>
#include <SLGrid.h>

//-----------------------------------------------------------------------------
//! SLGrid ctor with min & max corners and its resolutions
SLGrid::SLGrid(SLAssetManager* assetMgr,
               SLVec3f         minXZ,
               SLVec3f         maxXZ,
               SLuint          resX,
               SLuint          resZ,
               SLstring        name,
               SLMaterial*     mat) : SLMesh(assetMgr, name)
{
    assert(minXZ != maxXZ);
    assert(resX > 0);
    assert(resZ > 0);
    assert(name != "");

    _primitive = PT_lines;
    _min       = minXZ;
    _max       = maxXZ;
    _resX      = resX;
    _resZ      = resZ;
    _isVolume  = false;

    buildMesh(mat);
}
//-----------------------------------------------------------------------------
//! SLGrid::buildMesh fills in the underlying arrays from the SLMesh object
void SLGrid::buildMesh(SLMaterial* material)
{
    deleteData();

    // Check max. allowed no. of verts
    SLuint64 uIntNumV64 = (_resX + 1) * 2 + (_resZ + 1) * 2;
    if (uIntNumV64 > UINT_MAX)
        SL_EXIT_MSG("SLGrid supports max. 2^32 vertices.");

    // allocate vectors of SLMesh
    P.clear();
    P.resize((_resX + 1) * 2 + (_resZ + 1) * 2);

    if (uIntNumV64 < 65535)
    {
        I16.clear();
        I16.resize(P.size());
    }
    else
    {
        I32.clear();
        I32.resize(P.size());
    }

    // Set one default material index
    mat(material);

    // delta vector
    SLVec3f d = _max - _min;
    d.x /= (SLfloat)_resX;
    d.z /= (SLfloat)_resZ;

    // Build vertex data
    SLuint p = 0;
    for (SLuint x = 0; x <= _resX; ++x)
    {
        P[p++].set(_min.x + x * d.x, 0, _min.z);
        P[p++].set(_min.x + x * d.x, 0, _max.z);
    }
    for (SLuint z = 0; z <= _resZ; ++z)
    {
        P[p++].set(_min.x, 0, _min.z + z * d.z);
        P[p++].set(_max.x, 0, _min.z + z * d.z);
    }

    // Indexes
    if (I16.size())
        for (SLushort i = 0; i < P.size(); ++i)
            I16[i] = i;
    else
        for (SLuint i = 0; i < P.size(); ++i)
            I32[i] = i;
}
//-----------------------------------------------------------------------------
