//#############################################################################
//  File:      SLTriangle.cpp
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Philipp Jueni, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLTriangle.h>

//-----------------------------------------------------------------------------
SLTriangle::SLTriangle(SLAssetManager* assetMgr,
                       SLMaterial*     material,
                       const SLstring& name,
                       const SLVec3f&  p0,
                       const SLVec3f&  p1,
                       const SLVec3f&  p2,
                       const SLVec2f&  t0,
                       const SLVec2f&  t1,
                       const SLVec2f&  t2) : SLMesh(assetMgr, name)
{
    p[0] = p0;
    p[1] = p1;
    p[2] = p2;

    t[0] = t0;
    t[1] = t1;
    t[2] = t2;

    mat(material);

    _isVolume = false;

    buildMesh(material);
}
//-----------------------------------------------------------------------------
//! Builds the mesh by copying the vertex info into the arrays of SLMescj
void SLTriangle::buildMesh(SLMaterial* material)
{
    deleteData();

    P.clear();
    P.resize(3);            // Vector for positions
    N.clear();
    N.resize(P.size());     // Vector for vertex normals (opt.)
    UV[0].clear();
    UV[0].resize(P.size()); // Vector for vertex tex. coords. (opt.)
    UV[1].clear();
    I16.clear();
    I16.resize(3); // Vector for vertex indices 16 bit

    // vertex positions
    P[0] = p[0];
    P[1] = p[1];
    P[2] = p[2];

    // vertex texture coordinates
    UV[0][0] = t[0];
    UV[0][1] = t[1];
    UV[0][2] = t[2];

    // indices
    I16[0] = 0;
    I16[1] = 1;
    I16[2] = 2;

    // normals
    SLVec3f n = (p[1] - p[0]) ^ (p[2] - p[0]);
    n.normalize();
    N[0] = n;
    N[1] = n;
    N[2] = n;
}
//-----------------------------------------------------------------------------
