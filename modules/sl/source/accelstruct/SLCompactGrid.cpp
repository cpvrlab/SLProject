//#############################################################################
//  File:      SLCompactGrid.cpp
//  Authors:   Manuel Frischknecht, Marcus Hudritsch
//  Date:      July 2015
//  Authors:   Manuel Frischknecht, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLCompactGrid.h>
#include <SLNode.h>
#include <SLRay.h>
#include <Moeller/TriangleBoxIntersect.h>
#include <Profiler.h>

//-----------------------------------------------------------------------------
SLCompactGrid::SLCompactGrid(SLMesh* m) : SLAccelStruct(m)
{
    _voxelCnt      = 0;
    _voxelCntEmpty = 0;
    _voxelMaxTria  = 0;
}
//-----------------------------------------------------------------------------
//! Returns the indices of the voxel around a given point
SLVec3i SLCompactGrid::containingVoxel(const SLVec3f& p) const
{
    SLVec3i pos;
    SLVec3f delta = p - _minV;
    pos.x         = (SLint)(delta.x / _voxelSize.x);
    pos.y         = (SLint)(delta.y / _voxelSize.y);
    pos.z         = (SLint)(delta.z / _voxelSize.z);

    // Check bounds of voxel indices
    if (pos.x >= (SLint)_size.x) pos.x = (SLint)_size.x - 1;
    if (pos.x < 0) pos.x = 0;
    if (pos.y >= (SLint)_size.y) pos.y = (SLint)_size.y - 1;
    if (pos.y < 0) pos.y = 0;
    if (pos.z >= (SLint)_size.z) pos.z = (SLint)_size.z - 1;
    if (pos.z < 0) pos.z = 0;

    return pos;
}
//-----------------------------------------------------------------------------
//! Returns the voxel center point for a given voxel by index
SLVec3f SLCompactGrid::voxelCenter(const SLVec3i& pos) const
{
    return _minV + SLVec3f((pos.x + .5f) * _voxelSize.x,
                           (pos.y + .5f) * _voxelSize.y,
                           (pos.z + .5f) * _voxelSize.z);
}
//-----------------------------------------------------------------------------
//! Returns the min. and max. voxel of a triangle
void SLCompactGrid::getMinMaxVoxel(const Triangle& triangle,
                                   SLVec3i&        minCell,
                                   SLVec3i&        maxCell)
{
    minCell = maxCell = containingVoxel(triangle[0]);
    for (SLuint i = 1; i < 3; ++i)
    {
        auto& vertex = triangle[i];
        minCell.setMin(containingVoxel(vertex));
        maxCell.setMax(containingVoxel(vertex));
    }
}
//-----------------------------------------------------------------------------
//! Deletes the entire uniform grid data
void SLCompactGrid::deleteAll()
{
    _voxelCnt      = 0;
    _voxelCntEmpty = 0;
    _voxelMaxTria  = 0;
    _voxelAvgTria  = 0;

    _voxelOffsets.clear();
    _triangleIndexes16.clear();
    _triangleIndexes32.clear();

    disposeBuffers();
}
//-----------------------------------------------------------------------------
//! Loops over triangles gets their voxels and calls the callback function
void SLCompactGrid::ifTriangleInVoxelDo(triVoxCallback callback)
{
    assert(callback && "No callback function passed");

    for (SLuint i = 0; i < _numTriangles; ++i)
    {
        auto index = [&](SLuint j)
        { return _m->I16.size()
                   ? _m->I16[i * 3 + j]
                   : _m->I32[i * 3 + j]; };
        Triangle triangle = {_m->finalP(index(0)),
                             _m->finalP(index(1)),
                             _m->finalP(index(2))};
        SLVec3i  min, max, pos;
        getMinMaxVoxel(triangle, min, max);

        for (pos.z = min.z; pos.z <= max.z; ++pos.z)
        {
            for (pos.y = min.y; pos.y <= max.y; ++pos.y)
            {
                for (pos.x = min.x; pos.x <= max.x; ++pos.x)
                {
                    SLuint  voxIndex  = indexAtPos(pos);
                    SLVec3f voxCenter = voxelCenter(pos);
                    if (triBoxOverlap(*((float(*)[3]) & voxCenter),
                                      *((float(*)[3]) & _voxelSizeHalf),
                                      *((float(*)[3][3]) & triangle)))
                    {
                        callback(i, voxIndex);
                    }
                }
            }
        }
    }
}
//-----------------------------------------------------------------------------
/*!
SLCompactGrid::build implements the data structure proposed by Lagae & Dutre in
their paper "Compact, Fast and Robust Grids for Ray Tracing".
*/
void SLCompactGrid::build(SLVec3f minV, SLVec3f maxV)
{
    PROFILE_FUNCTION();

    assert(_m->I16.size() || _m->I32.size());

    deleteAll();

    _minV         = minV;
    _maxV         = maxV;
    _numTriangles = _m->numI() / 3;

    // Calculate grid size
    const SLfloat DENSITY = 8;
    SLVec3f       size    = _maxV - _minV;
    SLfloat       volume  = size.x * size.y * size.z;
    if (volume < FLT_EPSILON)
    {
        SL_WARN_MSG("\n\n **** SLCompactGrid::build: Zero Volume. ****");
        return;
    }

    float f        = cbrtf(DENSITY * _numTriangles / volume);
    _voxelSize.x   = size.x / ceil(size.x * f);
    _voxelSize.y   = size.y / ceil(size.y * f);
    _voxelSize.z   = size.z / ceil(size.z * f);
    _voxelSizeHalf = _voxelSize * 0.5f;
    _size.x        = (SLuint)ceil(size.x / _voxelSize.x);
    _size.y        = (SLuint)ceil(size.y / _voxelSize.y);
    _size.z        = (SLuint)ceil(size.z / _voxelSize.z);
    _voxelCnt      = _size.x * _size.y * _size.z;
    _voxelOffsets.assign(_voxelCnt + 1, 0);

    ifTriangleInVoxelDo([&](const SLuint& i, const SLuint& voxIndex)
                        { ++_voxelOffsets[voxIndex]; });

    // The last counter doesn't count and is always empty.
    _voxelMaxTria  = _voxelOffsets[0];
    _voxelCntEmpty = (_voxelOffsets[0] == 0) - 1;
    for (SLuint i = 1; i < _voxelOffsets.size(); ++i)
    {
        _voxelMaxTria = std::max(_voxelMaxTria, (SLuint)_voxelOffsets[i]);
        _voxelCntEmpty += _voxelOffsets[i] == 0;
        _voxelOffsets[i] += _voxelOffsets[i - 1];
    }

    if (_m->I16.size())
    {
        _triangleIndexes16.resize(_voxelOffsets.back());
        ifTriangleInVoxelDo([&](const SLushort& i, const SLuint& voxIndex)
                            {
            SLuint location              = --_voxelOffsets[voxIndex];
            _triangleIndexes16[location] = i; });
        _triangleIndexes16.shrink_to_fit();
    }
    else
    {
        _triangleIndexes32.resize(_voxelOffsets.back());
        ifTriangleInVoxelDo([&](const SLuint& i, const SLuint& voxIndex)
                            {
            SLuint location              = --_voxelOffsets[voxIndex];
            _triangleIndexes32[location] = i; });
        _triangleIndexes32.shrink_to_fit();
    }

    _voxelOffsets.shrink_to_fit();
}
//-----------------------------------------------------------------------------
//! Updates the statistics in the parent node
void SLCompactGrid::updateStats(SLNodeStats& stats)
{
    stats.numVoxels += _voxelCnt;
    stats.numVoxEmpty += _voxelCntEmpty;

    stats.numBytesAccel += sizeof(SLCompactGrid);
    stats.numBytesAccel += SL_sizeOfVector(_voxelOffsets);
    stats.numBytesAccel += _m->I16.size()
                             ? SL_sizeOfVector(_triangleIndexes16)
                             : SL_sizeOfVector(_triangleIndexes32);

    stats.numVoxMaxTria = std::max(_voxelMaxTria, stats.numVoxMaxTria);
}
//-----------------------------------------------------------------------------
//! SLCompactGrid::draw draws the non-empty voxels of the uniform grid
void SLCompactGrid::draw(SLSceneView* sv)
{
    if (_voxelCnt > 0)
    {
        if (!_vao.vaoID())
        {
            SLuint   x, y, z;
            SLuint   curVoxel = 0;
            SLVec3f  v;
            SLVVec3f P;

            // Loop through voxels
            v.z = _minV.z;
            for (z = 0; z < _size.z; ++z, v.z += _voxelSize.z)
            {
                v.y = _minV.y;
                for (y = 0; y < _size.y; ++y, v.y += _voxelSize.y)
                {
                    v.x = _minV.x;
                    for (x = 0; x < _size.x; ++x, v.x += _voxelSize.x)
                    {
                        SLuint voxelID = indexAtPos(SLVec3i((SLint)x, (SLint)y, (SLint)z));

                        if (_voxelOffsets[voxelID] < _voxelOffsets[voxelID + 1])
                        {
                            P.push_back(SLVec3f(v.x, v.y, v.z));
                            P.push_back(SLVec3f(v.x + _voxelSize.x, v.y, v.z));
                            P.push_back(SLVec3f(v.x + _voxelSize.x, v.y, v.z));
                            P.push_back(SLVec3f(v.x + _voxelSize.x, v.y, v.z + _voxelSize.z));
                            P.push_back(SLVec3f(v.x + _voxelSize.x, v.y, v.z + _voxelSize.z));
                            P.push_back(SLVec3f(v.x, v.y, v.z + _voxelSize.z));
                            P.push_back(SLVec3f(v.x, v.y, v.z + _voxelSize.z));
                            P.push_back(SLVec3f(v.x, v.y, v.z));

                            P.push_back(SLVec3f(v.x, v.y + _voxelSize.y, v.z));
                            P.push_back(SLVec3f(v.x + _voxelSize.x, v.y + _voxelSize.y, v.z));
                            P.push_back(SLVec3f(v.x + _voxelSize.x, v.y + _voxelSize.y, v.z));
                            P.push_back(SLVec3f(v.x + _voxelSize.x, v.y + _voxelSize.y, v.z + _voxelSize.z));
                            P.push_back(SLVec3f(v.x + _voxelSize.x, v.y + _voxelSize.y, v.z + _voxelSize.z));
                            P.push_back(SLVec3f(v.x, v.y + _voxelSize.y, v.z + _voxelSize.z));
                            P.push_back(SLVec3f(v.x, v.y + _voxelSize.y, v.z + _voxelSize.z));
                            P.push_back(SLVec3f(v.x, v.y + _voxelSize.y, v.z));

                            P.push_back(SLVec3f(v.x, v.y, v.z));
                            P.push_back(SLVec3f(v.x, v.y + _voxelSize.y, v.z));
                            P.push_back(SLVec3f(v.x + _voxelSize.x, v.y, v.z));
                            P.push_back(SLVec3f(v.x + _voxelSize.x, v.y + _voxelSize.y, v.z));
                            P.push_back(SLVec3f(v.x + _voxelSize.x, v.y, v.z + _voxelSize.z));
                            P.push_back(SLVec3f(v.x + _voxelSize.x, v.y + _voxelSize.y, v.z + _voxelSize.z));
                            P.push_back(SLVec3f(v.x, v.y, v.z + _voxelSize.z));
                            P.push_back(SLVec3f(v.x, v.y + _voxelSize.y, v.z + _voxelSize.z));
                        }
                        curVoxel++;
                    }
                }
            }

            _vao.generateVertexPos(&P);
        }

        _vao.drawArrayAsColored(PT_lines, SLCol4f::CYAN);
    }
}
//-----------------------------------------------------------------------------
/*!
Ray Mesh intersection method using the regular grid space subdivision structure
and a voxel traversal algorithm described in "A Fast Voxel Traversal Algorithm
for Ray Tracing" by John Amanatides and Andrew Woo.
*/
SLbool SLCompactGrid::intersect(SLRay* ray, SLNode* node)
{
    // Check first if the AABB is hit at all
    if (node->aabb()->isHitInOS(ray))
    {
        SLbool wasHit = false;

        if (_voxelCnt > 0)
        { // Calculate the intersection point with the AABB
            SLVec3f O          = ray->originOS;
            SLVec3f D          = ray->dirOS;
            SLVec3f invD       = ray->invDirOS;
            SLVec3f startPoint = O;

            ////Determine start voxel of the grid
            if (ray->tmin > 0) startPoint += ray->tmin * D;
            SLVec3i startVox = containingVoxel(startPoint);

            // Calculate the voxel ID into our 1D-voxel array
            SLuint voxID = indexAtPos(startVox);

            // Calculate steps: -1 or 1 on each axis
            // clang-format off
            SLint stepX = (D.x > 0) ? 1 : (D.x < 0) ? -1 : 0;
            SLint stepY = (D.y > 0) ? 1 : (D.y < 0) ? -1 : 0;
            SLint stepZ = (D.z > 0) ? 1 : (D.z < 0) ? -1 : 0;

            // Calculate the min. & max point of the start voxel
            SLVec3f minVox(_minV.x + startVox.x * _voxelSize.x,
                           _minV.y + startVox.y * _voxelSize.y,
                           _minV.z + startVox.z * _voxelSize.z);
            SLVec3f maxVox(minVox + _voxelSize);

            // Calculate max. dist along the ray for each component in tMaxX,Y,Z
            SLfloat tMaxX = FLT_MAX, tMaxY = FLT_MAX, tMaxZ = FLT_MAX;
            if (stepX ==  1) tMaxX = (maxVox.x - O.x) * invD.x; else
            if (stepX == -1) tMaxX = (minVox.x - O.x) * invD.x;
            if (stepY ==  1) tMaxY = (maxVox.y - O.y) * invD.y; else
            if (stepY == -1) tMaxY = (minVox.y - O.y) * invD.y;
            if (stepZ ==  1) tMaxZ = (maxVox.z - O.z) * invD.z; else
            if (stepZ == -1) tMaxZ = (minVox.z - O.z) * invD.z;
            // clang-format on

            // tMax is max. distance along the ray to stay in the current voxel
            SLfloat tMax = std::min(tMaxX, std::min(tMaxY, tMaxZ));

            // Precalculate the voxel id increment
            SLint incIDX = stepX;
            SLint incIDY = stepY * (SLint)_size.x;
            SLint incIDZ = stepZ * (SLint)_size.x * (SLint)_size.y;

            // Calculate tDeltaX,Y & Z (=dist. along the ray in a voxel)
            SLfloat tDeltaX = (_voxelSize.x * invD.x) * stepX;
            SLfloat tDeltaY = (_voxelSize.y * invD.y) * stepY;
            SLfloat tDeltaZ = (_voxelSize.z * invD.z) * stepZ;

            // Now traverse the voxels
            while (!wasHit)
            {
                if (_m->I16.size())
                {
                    for (SLuint i = _voxelOffsets[voxID]; i < _voxelOffsets[voxID + 1]; ++i)
                    {
                        if (_m->hitTriangleOS(ray, node, _triangleIndexes16[i] * 3))
                        {
                            if (ray->length <= tMax && !wasHit)
                                wasHit = true;
                        }
                    }
                }
                else
                {
                    for (SLuint i = _voxelOffsets[voxID]; i < _voxelOffsets[voxID + 1]; ++i)
                    {
                        if (_m->hitTriangleOS(ray, node, _triangleIndexes32[i] * 3))
                        {
                            if (ray->length <= tMax && !wasHit)
                                wasHit = true;
                        }
                    }
                }

                // step Voxel
                if (tMaxX < tMaxY)
                {
                    if (tMaxX < tMaxZ)
                    {
                        startVox.x += stepX;
                        if (startVox.x >= (SLint)_size.x || startVox.x < 0) return wasHit;
                        tMaxX += tDeltaX;
                        voxID += (SLuint)incIDX;
                        tMax = tMaxX;
                    }
                    else
                    {
                        startVox.z += stepZ;
                        if (startVox.z >= (SLint)_size.z || startVox.z < 0) return wasHit;
                        tMaxZ += tDeltaZ;
                        voxID += (SLuint)incIDZ;
                        tMax = tMaxZ;
                    }
                }
                else
                {
                    if (tMaxY < tMaxZ)
                    {
                        startVox.y += stepY;
                        if (startVox.y >= (SLint)_size.y || startVox.y < 0) return wasHit;
                        tMaxY += tDeltaY;
                        voxID += (SLuint)incIDY;
                        tMax = tMaxY;
                    }
                    else
                    {
                        startVox.z += stepZ;
                        if (startVox.z >= (SLint)_size.z || startVox.z < 0) return wasHit;
                        tMaxZ += tDeltaZ;
                        voxID += (SLuint)incIDZ;
                        tMax = tMaxZ;
                    }
                }
            }
            return wasHit;
        }
        else
        { // not enough triangles for regular grid > check them all
            for (SLuint t = 0; t < _m->numI(); t += 3)
            {
                if (_m->hitTriangleOS(ray, node, t) && !wasHit) wasHit = true;
            }
            return wasHit;
        }
    }
    else
        return false; // did not hit aabb
}
//-----------------------------------------------------------------------------
