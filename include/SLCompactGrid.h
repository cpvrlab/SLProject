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

#include <stdafx.h>

#include <array>
#include <vector>
#include <numeric>
#include <atomic>

#include <SLVec3.h>
#include <SLAccelStruct.h>

//-----------------------------------------------------------------------------
class SLCompactGrid : public SLAccelStruct
{
    public:
    using Triangle = std::array<SLVec3f,3>;

    /*
    template<class T>
    struct Counter
    {
		Counter() = default;
        inline Counter(T i): count(i) {}
        inline Counter(const Counter &c):
            count(c.count.load())
        {}

        inline Counter &operator=(const Counter &c)
        {
            count = c.count.load();
            return *this;
        }

        std::atomic<T> count;
    };
    */
                            SLCompactGrid   (SLMesh* m);
                           ~SLCompactGrid   (){;}

                void        build           (SLVec3f minV, SLVec3f maxV);
                void        updateStats     (SLNodeStats &stats);
                void        draw            (SLSceneView* sv);
                SLbool      intersect       (SLRay* ray, SLNode* node);

				void        deleteAll       (){}
                void        disposeBuffers(){}

                SLint       indexAtPos      (const SLVec3i &p) const 
                                            { return p.x + p.y * _size.x + p.z * _size.x * _size.y; }
                SLVec3i     containingCell  (const SLVec3f &p) const;
                void        setMinMaxCell   (const Triangle &triangle, 
                                             SLVec3i &minCell, 
                                             SLVec3i &maxCell);

    private:

                std::vector<SLuint> _cellOffsets;
                SLVuint _triangleIndices;

                SLVec3i _size;                  //!< num. of cells of grid
                SLfloat _cellWidth;             //!< cell width
                SLuint _numVoxels;
                SLuint _emptyVoxels;
                SLuint _maxTrianglesPerVoxel;
};
//-----------------------------------------------------------------------------
#endif //SL_COMPACTGRID
