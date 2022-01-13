//#############################################################################
//  File:      SLPolygon.cpp
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLPolygon.h>

//-----------------------------------------------------------------------------
//! SLPolygon ctor with corner points vector
SLPolygon::SLPolygon(SLAssetManager* assetMgr,
                     const SLVVec3f& corners,
                     const SLstring& name,
                     SLMaterial*     mat)
  : SLMesh(assetMgr, name)
{
    assert(corners.size() > 2);
    _corners = corners;
    buildMesh(mat);
}
//-----------------------------------------------------------------------------
//! SLPolygon ctor with corner points and its texture coords vector
SLPolygon::SLPolygon(SLAssetManager* assetMgr,
                     const SLVVec3f& corners,
                     const SLVVec2f& texCoords,
                     const SLstring& name,
                     SLMaterial*     mat) : SLMesh(assetMgr, name)
{
    assert(corners.size() > 2 && texCoords.size() == corners.size());
    _corners = corners;
    _uv1     = texCoords;
    buildMesh(mat);
}
//-----------------------------------------------------------------------------
//! SLPolygon ctor for centered light rectangle in x/y-plane w. N=-z
SLPolygon::SLPolygon(SLAssetManager* assetMgr,
                     SLfloat         width,
                     SLfloat         height,
                     const SLstring& name,
                     SLMaterial*     mat) : SLMesh(assetMgr, name)
{
    assert(width > 0 && height > 0);
    SLfloat hw = width * 0.5f;
    SLfloat hh = height * 0.5f;
    _corners.push_back(SLVec3f(hw, hh));
    _corners.push_back(SLVec3f(hw, -hh));
    _corners.push_back(SLVec3f(-hw, -hh));
    _corners.push_back(SLVec3f(-hw, hh));
    buildMesh(mat);
}
//-----------------------------------------------------------------------------
//! SLPolygon::buildMesh fills in the underlying arrays from the SLMesh object
void SLPolygon::buildMesh(SLMaterial* material)
{
    _isVolume = false;

    deleteData();

    // Check max. allowed no. of verts
    if (_corners.size() >= 65535)
        SL_EXIT_MSG("SLPolygon::buildMesh: NO. of vertices exceeds the maximum (65535) allowed.");

    // allocate vectors of SLMesh
    P.clear();
    P.resize(_corners.size());
    N.clear();
    N.resize(P.size());
    if (!_uv1.empty())
    {
        UV[0].clear();
        UV[0].resize(P.size());
    }
    UV[1].clear();
    I16.clear();
    I16.resize((P.size() - 2) * 3);

    // Calculate normal from the first 3 corners
    SLVec3f v1(_corners[0] - _corners[1]);
    SLVec3f v2(_corners[0] - _corners[2]);
    SLVec3f n(v1 ^ v2);
    n.normalize();

    // Set one default material index
    mat(material);

    // Copy vertices and normals
    for (SLushort i = 0; i < (SLushort)P.size(); ++i)
    {
        P[i] = _corners[i];
        N[i] = n;
        if (!UV[0].empty()) UV[0][i] = _uv1[i];
    }

    // Build face vertex indices
    for (SLushort f = 0; f < (SLushort)_corners.size() - 2; ++f)
    {
        SLuint i   = f * 3;
        I16[i]     = 0;
        I16[i + 1] = f + 1;
        I16[i + 2] = f + 2;
    }
}
//-----------------------------------------------------------------------------
