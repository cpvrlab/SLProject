//#############################################################################
//  File:      SLRectangle.cpp
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLRectangle.h>
#include <limits.h>
#include <utility>

//-----------------------------------------------------------------------------
//! SLRectangle ctor with min & max corners and its resolutions
SLRectangle::SLRectangle(SLAssetManager* assetMgr,
                         const SLVec2f&  min,
                         const SLVec2f&  max,
                         SLuint          resX,
                         SLuint          resY,
                         const SLstring& name,
                         SLMaterial*     mat) : SLMesh(assetMgr, name)
{
    assert(min != max);
    assert(resX > 0);
    assert(resY > 0);
    assert(!name.empty());
    _min = min;
    _max = max;
    _uv_min.set(0, 0);
    _uv_max.set(1, 1);
    _resX     = resX;
    _resY     = resY;
    _isVolume = true;
    buildMesh(mat);
}
//-----------------------------------------------------------------------------
//! SLRectangle ctor with min & max corners and its resolutions
SLRectangle::SLRectangle(SLAssetManager* assetMgr,
                         const SLVec2f&  min,
                         const SLVec2f&  max,
                         const SLVec2f&  uv_min,
                         const SLVec2f&  uv_max,
                         SLuint          resX,
                         SLuint          resY,
                         const SLstring& name,
                         SLMaterial*     mat) : SLMesh(assetMgr, name)
{
    assert(min != max);
    assert(uv_min != uv_max);
    assert(resX > 0);
    assert(resY > 0);
    assert(!name.empty());
    _min      = min;
    _max      = max;
    _uv_min   = uv_min;
    _uv_max   = uv_max;
    _resX     = resX;
    _resY     = resY;
    _isVolume = true;
    buildMesh(mat);
}
//-----------------------------------------------------------------------------
//! SLRectangle::buildMesh fills in the underlying arrays from the SLMesh object
void SLRectangle::buildMesh(SLMaterial* material)
{
    deleteData();

    // Check max. allowed no. of vertices
    SLuint uIntNumV64 = (_resX + 1) * (_resY + 1);
    if (uIntNumV64 > UINT_MAX)
        SL_EXIT_MSG("SLMesh supports max. 2^32 vertices.");

    // allocate vectors of SLMesh
    P.clear();
    P.resize((_resX + 1) * (_resY + 1));
    N.clear();
    N.resize(P.size());
    UV[0].clear();
    UV[0].resize(P.size());
    UV[1].clear();

    if (uIntNumV64 < 65535)
    {
        I16.clear();
        I16.resize(_resX * _resY * 2 * 3);
    }
    else
    {
        I32.clear();
        I32.resize(_resX * _resY * 2 * 3);
    }

    // Calculate normal from the first 3 corners
    SLVec3f maxmin(_max.x, _min.y, 0);
    SLVec3f minmax(_min.x, _max.y, 0);
    SLVec3f e1(maxmin - _min);
    SLVec3f e2(minmax - _min);
    SLVec3f curN(e1 ^ e2);
    curN.normalize();

    // Set one default material index
    mat(material);

    // define delta vectors dX & dY and deltas for tex. coord. dU,dV
    SLVec3f dX = e1 / (SLfloat)_resX;
    SLVec3f dY = e2 / (SLfloat)_resY;
    SLfloat dU = (_uv_max.x - _uv_min.x) / (SLfloat)_resX;
    SLfloat dV = (_uv_max.y - _uv_min.y) / (SLfloat)_resY;

    // Build vertex data
    SLuint i = 0;
    for (SLuint y = 0; y <= _resY; ++y)
    {
        SLVec3f curV = _min;
        SLVec2f curT = _uv_min;
        curV += (SLfloat)y * dY;
        curT.y += (SLfloat)y * dV;

        for (SLuint x = 0; x <= _resX; ++x, ++i)
        {
            P[i]     = curV;
            UV[0][i] = curT;
            N[i]     = curN;
            curV += dX;
            curT.x += dU;
        }
    }

    // Build face vertex indices
    if (!I16.empty())
    {
        SLushort v = 0, i = 0; // index for vertices and indices
        for (SLuint y = 0; y < _resY; ++y)
        {
            for (SLuint x = 0; x < _resX; ++x, ++v)
            { // triangle 1
                I16[i++] = v;
                I16[i++] = v + (SLushort)_resX + 2;
                I16[i++] = v + (SLushort)_resX + 1;

                // triangle 2
                I16[i++] = v;
                I16[i++] = v + 1;
                I16[i++] = v + (SLushort)_resX + 2;
            }
            v++;
        }
    }
    else
    {
        SLuint v = 0, i = 0; // index for vertices and indices
        for (SLuint y = 0; y < _resY; ++y)
        {
            for (SLuint x = 0; x < _resX; ++x, ++v)
            { // triangle 1
                I32[i++] = v;
                I32[i++] = v + _resX + 2;
                I32[i++] = v + _resX + 1;

                // triangle 2
                I32[i++] = v;
                I32[i++] = v + 1;
                I32[i++] = v + _resX + 2;
            }
            v++;
        }
    }
}
//-----------------------------------------------------------------------------
