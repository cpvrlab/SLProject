//#############################################################################
//  File:      SLAccelStruct.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLACCELSTRUCT_H
#define SLACCELSTRUCT_H

#include <stdafx.h>
#include <SLMesh.h>

//-----------------------------------------------------------------------------
//! SLAccelStruct is an abstract base class for acceleration structures
/*! The SLAccelStruct class serves as common class for the SLUniformGrid and 
the SLKDTree class. All derived acceleration structures must be able to build,
draw, intersect with a ray and update statistics. All structures work on meshes. 
*/
class SLAccelStruct
{
    public:
                               SLAccelStruct (SLMesh* m){_m=m;}
        virtual               ~SLAccelStruct (){;}

        virtual void           build          (SLVec3f minV, SLVec3f maxV) = 0;
        virtual void           updateStats    (SLNodeStats &stats) = 0;
        virtual void           draw           (SLSceneView* sv) = 0;
        virtual SLbool         intersect      (SLRay* ray, SLNode* node) = 0;
        virtual void           disposeBuffers () = 0;
   
    protected:
                SLMesh*        _m;            //!< Pointer to the mesh
                SLVec3f        _minV;         //!< min. point of AABB
                SLVec3f        _maxV;         //!< max. point of AABB
               
                SLuint         _voxCnt;       //!< Num. of voxels in accelerator
                SLuint         _voxCntEmpty;  //!< Num. of empty voxels
                SLuint         _voxMaxTria;   //!< max. num. of triangles pre voxel
                SLfloat        _voxAvgTria;   //!< avg. num. of triangles per voxel

};
//-----------------------------------------------------------------------------
#endif //SLACCELSTRUCT_H

