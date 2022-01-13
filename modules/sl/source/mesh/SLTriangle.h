//#############################################################################
//  File:      SLTriangle.h
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Philipp Jueni, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLTRIANGLE_H
#define SLTRIANGLE_H

#include <SLMesh.h>

//-----------------------------------------------------------------------------
//! A triangle class as the most simplest mesh
class SLTriangle : public SLMesh
{
public:
    SLTriangle(SLAssetManager* assetMgr,
               SLMaterial*     mat,
               const SLstring& name = "triangle mesh",
               const SLVec3f&  p0   = SLVec3f(0, 0, 0),
               const SLVec3f&  p1   = SLVec3f(1, 0, 0),
               const SLVec3f&  p2   = SLVec3f(0, 1, 0),
               const SLVec2f&  t0   = SLVec2f(0, 0),
               const SLVec2f&  t1   = SLVec2f(1, 0),
               const SLVec2f&  t2   = SLVec2f(0, 1));

    void buildMesh(SLMaterial* mat);

protected:
    SLVec3f p[3]; //!< Array of 3 vertex positions
    SLVec2f t[3]; //!< Array of 3 vertex tex. coords. (opt.)
};
//-----------------------------------------------------------------------------
#endif // SLTRIANGLE_H