//#############################################################################
//  File:      SLCompactGrid.h
//  Author:    Manuel Frischknecht, Marcus Hudritsch
//  Date:      July 2015
//  Copyright: Manuel Frischknecht, Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SL_COMPACTGRID
#define SL_COMPACTGRID

#include <array>
#include <vector>
#include <numeric>
#include <atomic>

#include <SLVec3.h>
#include <SLAccelStruct.h>
#include <SLGLVertexArrayExt.h>

//-----------------------------------------------------------------------------
typedef std::function<void(const SLuint, const SLuint)> triVoxCallback;
//-----------------------------------------------------------------------------
//! Class for compact uniform grid acceleration structure
/*! This class implements the data structure proposed by Lagae & Dutre in their
paper "Compact, Fast and Robust Grids for Ray Tracing". It reduces the memory
footprint to 20% of a regular uniform grid implemented in SLUniformGrid.
*/
class SLCompactGrid : public SLAccelStruct
{
    public:
    using Triangle = std::array<SLVec3f,3>;

                    SLCompactGrid       (SLMesh* m);
                    ~SLCompactGrid       (){;}

        void        build               (SLVec3f minV, SLVec3f maxV);
        void        updateStats         (SLNodeStats &stats);
        void        draw                (SLSceneView* sv);
        SLbool      intersect           (SLRay* ray, SLNode* node);
                
        void        deleteAll           ();
        void        disposeBuffers      (){if (_vao.id()) _vao.clearAttribs();}

        SLuint      indexAtPos          (const SLVec3i &p) const 
                                        {return p.x + p.y * _size.x + 
                                                p.z*_size.x * _size.y;}
        SLVec3f     voxelCenter         (const SLVec3i &pos) const;
        SLVec3i     containingVoxel     (const SLVec3f &p) const;
        void        getMinMaxVoxel      (const Triangle &triangle, 
                                            SLVec3i &minCell, 
                                            SLVec3i &maxCell);
        void        ifTriangleInVoxelDo (triVoxCallback cb);
    private:
        SLVec3i     _size;              //!< num. of voxel in grid dir.
        SLuint      _numTriangles;      //!< NO. of triangles in the mesh
        SLVec3f     _voxelSize;         //!< size of a voxel
        SLVec3f     _voxelSizeHalf;     //!< half size of a voxel
        SLVuint     _voxelOffsets;      //!< Offset array (C in the paper)
        SLVushort   _triangleIndexes16; //!< 16 bit triangle index array (L in the paper)
        SLVuint     _triangleIndexes32; //!< 32 bit triangle index array (L in the paper)
        SLGLVertexArrayExt  _vao;       //!< Vertex array object for rendering
};
//-----------------------------------------------------------------------------
#endif //SL_COMPACTGRID
