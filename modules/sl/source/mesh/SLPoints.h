//#############################################################################
//  File:      SLPoints.h
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPOINTS_H
#define SLPOINTS_H

#include <SLMesh.h>
#include <SLRnd3f.h>

//-----------------------------------------------------------------------------
//! SLPoints creates
/*! The SLPoints mesh object of witch the vertices are drawn as points.
 */
class SLPoints : public SLMesh
{
public:
    //! Ctor for a given vector of points
    SLPoints(SLAssetManager* assetMgr,
             const SLVVec3f& points,
             const SLstring& name     = "point cloud",
             SLMaterial*     material = nullptr);
    SLPoints(SLAssetManager* assetMgr,
             const SLVVec3f& points,
             const SLVVec3f& normals,
             const SLstring& name     = "point cloud",
             SLMaterial*     material = 0);

    //! Ctor for a random point cloud.
    SLPoints(SLAssetManager* assetMgr,
             SLfloat         nPoints,
             SLRnd3f&        rnd,
             const SLstring& name = "normal point cloud",
             SLMaterial*     mat  = nullptr);
};
//-----------------------------------------------------------------------------
#endif
