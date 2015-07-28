//#############################################################################
//  File:      SLUniformGrid.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLUNIFORMGRID_H
#define SLUNIFORMGRID_H

#include <stdafx.h>

#include <SLAccelStruct.h>

//-----------------------------------------------------------------------------
//! Special vector type similar to std::vector w. smaller size type
typedef SLVector<SLuint, SLushort> SLV32ushort;
//-----------------------------------------------------------------------------
//! SLUniformGrid is an acceleration structure the uniformly subdivides space.
/*! A uniform grid is an axis aligned, regularly subdivided grid for 
accelerating the ray-triangle intersection.
*/
class SLUniformGrid : public SLAccelStruct    
{
    public:
                                SLUniformGrid (SLMesh* m);
                               ~SLUniformGrid ();

                void            build           (SLVec3f minV, SLVec3f maxV);
                void            updateStats     (SLNodeStats &stats);
                void            draw            (SLSceneView* sv);
                SLbool          intersect       (SLRay* ray, SLNode* node);
               
                void            deleteAll       ();
                void            disposeBuffers  (){if (_bufP.id()) _bufP.dispose();}
                   
    private:
                SLV32ushort**   _voxel;         //!< 1D voxel array for triangle indexes
                SLVec3i         _size;          //!< Grid size in x,y,z direction
                SLVec3f         _voxelSize;     //!< Voxel size
                SLGLBuffer      _bufP;          //!< Buffer object for vertex positions
};
//-----------------------------------------------------------------------------
#endif //SLUNIFORMGRID_H

