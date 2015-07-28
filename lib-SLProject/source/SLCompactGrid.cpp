//#############################################################################
//  File:      SLCompactGrid.cpp
//  Author:    Manuel Frischknecht, Marcus Hudritsch
//  Date:      July 2015
//  Copyright: Manuel Frischknecht, Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>

#include <SLCompactGrid.h>
#include <SLNode.h>
#include <SLRay.h>
#include <TriangleBoxIntersect.h>

//-----------------------------------------------------------------------------
SLCompactGrid::SLCompactGrid(SLMesh* m) : SLAccelStruct(m)
{
    _voxelCnt = 0;
    _voxelCntEmpty = 0; 
    _voxelMaxTria = 0;
    _voxelWidth = 0;
}
//-----------------------------------------------------------------------------
//! Returns the indices of the voxel around a given point
SLVec3i SLCompactGrid::containingVoxel(const SLVec3f &p) const
{
    SLVec3i pos;
    SLVec3f delta = p - _minV;
    pos.x = (SLint)(delta.x / _voxelWidth);
    pos.y = (SLint)(delta.y / _voxelWidth);
    pos.z = (SLint)(delta.z / _voxelWidth);

    // Check bounds of voxel indexes
    if (pos.x >= _size.x) pos.x = _size.x - 1; if (pos.x < 0) pos.x = 0;
    if (pos.y >= _size.y) pos.y = _size.y - 1; if (pos.y < 0) pos.y = 0;
    if (pos.z >= _size.z) pos.z = _size.z - 1; if (pos.z < 0) pos.z = 0;

    return pos;
}
//-----------------------------------------------------------------------------
//! Returns the voxel center point for a given voxel by index
SLVec3f SLCompactGrid::voxelCenter(const SLVec3i &pos) const
{
    return _minV + SLVec3f((pos.x + .5f)*_voxelWidth,
                           (pos.y + .5f)*_voxelWidth,
                           (pos.z + .5f)*_voxelWidth);
}
//-----------------------------------------------------------------------------
//! Returns the min. and max. voxel of a triangle
void SLCompactGrid::setMinMaxVoxel(const Triangle &triangle,
                                  SLVec3i &minCell,
                                  SLVec3i &maxCell)
{
    minCell = maxCell = containingVoxel(triangle[0]);
    for (int i = 1; i < 3; ++i)
    {   auto &vertex = triangle[i];
        minCell.setMin(containingVoxel(vertex));
        maxCell.setMax(containingVoxel(vertex));
    }
}
//-----------------------------------------------------------------------------
/*!
SLCompactGrid::build implements the data structure proposed by Lagae & Dutré in 
their paper "Compact, Fast and Robust Grids for Ray Tracing".
*/
void SLCompactGrid::build (SLVec3f minV, SLVec3f maxV)
{
    static const float DENSITY = 8;
    SLint numTriangles = _m->numI / 3;

    _minV = minV;
    _maxV = maxV;

    // Calculate grid size
    SLVec3f diagonal = _maxV - _minV;
    SLfloat volume = diagonal.x * diagonal.y * diagonal.z;
    float f = cbrtf(DENSITY*numTriangles / volume);
    _voxelWidth = SL_min(SL_min(diagonal.x / ceil(diagonal.x*f),
                                diagonal.y / ceil(diagonal.y*f)),
                                diagonal.z / ceil(diagonal.z*f));
    _size.x = (SLint)ceil(diagonal.x / _voxelWidth);
    _size.y = (SLint)ceil(diagonal.y / _voxelWidth);
    _size.z = (SLint)ceil(diagonal.z / _voxelWidth);

    SLVec3f voxHalfSize(_voxelWidth / 2, _voxelWidth / 2, _voxelWidth / 2);

    _voxelCnt = _size.x * _size.y * _size.z;
    _voxelOffsets.assign(_voxelCnt + 1, 0);


    for (int i = 0; i < numTriangles; ++i)
    {
        auto index  = [&](int j) { return _m->I16 ? _m->I16[i * 3 + j] : _m->I32[i * 3 + j]; };
        auto vertex = [&](int j) { return _m->finalP()[index(j)]; };
        Triangle triangle = { vertex(0), vertex(1), vertex(2) };
        SLVec3i min, max, pos;
        setMinMaxVoxel(triangle, min, max);

        for (pos.x = min.x; pos.x <= max.x; ++pos.x)
        {   for (pos.y = min.y; pos.y <= max.y; ++pos.y)
            {   for (pos.z = min.z; pos.z <= max.z; ++pos.z)
                {   
                    SLuint voxIndex = indexAtPos(pos);
                    SLVec3f voxCenter = voxelCenter(pos);
                    if (triBoxOverlap(*((float(*)[3])&voxCenter),
                                      *((float(*)[3])&voxHalfSize),
                                      *((float(*)[3][3])&triangle)))
                        ++_voxelOffsets[voxIndex];
                }
            }
        }
    }

    //The last counter doesn't count and is always empty.
    _voxelMaxTria = _voxelOffsets[0];
    _voxelCntEmpty = (_voxelOffsets[0] == 0) - 1;
    for (int i = 1; i < _voxelOffsets.size(); ++i)
    {   _voxelMaxTria = SL_max(_voxelMaxTria, (SLuint)_voxelOffsets[i]);
        _voxelCntEmpty += _voxelOffsets[i] == 0;
        _voxelOffsets[i] += _voxelOffsets[i - 1];
    }

    _triangleIndexes.resize(_voxelOffsets.back());


    // Reverse iterate over triangles
    for (int i = numTriangles - 1; i >= 0; --i)
    {
        auto index  = [&](int j) {return _m->I16 ? _m->I16[i * 3 + j] : _m->I32[i * 3 + j];};
        auto vertex = [&](int j) {return _m->finalP()[index(j)];};
        Triangle triangle = { vertex(0), vertex(1), vertex(2) };
        SLVec3i min, max, pos;
        setMinMaxVoxel(triangle, min, max);

        for (pos.x = min.x; pos.x <= max.x; ++pos.x)
        {   for (pos.y = min.y; pos.y <= max.y; ++pos.y)
            {   for (pos.z = min.z; pos.z <= max.z; ++pos.z)
                {   
                    SLuint voxIndex = indexAtPos(pos);
                    SLVec3f voxCenter = voxelCenter(pos);
                    if (triBoxOverlap(*((float(*)[3])&voxCenter),
                                      *((float(*)[3])&voxHalfSize),
                                      *((float(*)[3][3])&triangle)))
                    {   SLint location = --_voxelOffsets[voxIndex];
                        _triangleIndexes[location] = i;
                    }
                }
            }
        }
    }

    _voxelOffsets.shrink_to_fit();
    _triangleIndexes.shrink_to_fit();
}
//-----------------------------------------------------------------------------
void SLCompactGrid::updateStats (SLNodeStats &stats)
{
    stats.numVoxels += _voxelCnt;
    stats.numVoxEmpty += _voxelCntEmpty;

    stats.numBytesAccel += sizeof(SLCompactGrid);
    stats.numBytesAccel += SL_sizeOfVector(_voxelOffsets);
    stats.numBytesAccel += SL_sizeOfVector(_triangleIndexes);

    stats.numVoxMaxTria = SL_max(_voxelMaxTria, stats.numVoxMaxTria);
}
//-----------------------------------------------------------------------------
//! SLCompactGrid::draw draws the non-empty voxels of the uniform grid
void SLCompactGrid::draw (SLSceneView* sv)
{
    if (_voxelCnt > 0)
    {
        SLuint   i = 0;  // number of lines to draw

        if (!_bufP.id())
        {
            SLint    x, y, z;
            SLuint   curVoxel = 0;
            SLVec3f  v;
            SLuint   numP = 12 * 2 * _voxelCnt;
            SLVec3f* P = new SLVec3f[numP];
            SLVec3f diagonal = _maxV - _minV;
            SLfloat voxExtX = diagonal.x / _size.x;
            SLfloat voxExtY = diagonal.y / _size.y;
            SLfloat voxExtZ = diagonal.z / _size.z;

            // Loop through voxels
            v.z = _minV.z;
            for (z = 0; z < _size.z; ++z, v.z += voxExtZ)
            {
                v.y = _minV.y;
                for (y = 0; y < _size.y; ++y, v.y += voxExtY)
                {
                    v.x = _minV.x;
                    for (x = 0; x<_size.x; ++x, v.x += voxExtX)
                    {
                        SLuint voxelID = indexAtPos(SLVec3i(x,y,z));
                        
                        if (_voxelOffsets[voxelID] < _voxelOffsets[voxelID + 1])
                        {
                            P[i++].set(v.x, v.y, v.z);
                            P[i++].set(v.x + voxExtX, v.y, v.z);
                            P[i++].set(v.x + voxExtX, v.y, v.z);
                            P[i++].set(v.x + voxExtX, v.y, v.z + voxExtZ);
                            P[i++].set(v.x + voxExtX, v.y, v.z + voxExtZ);
                            P[i++].set(v.x, v.y, v.z + voxExtZ);
                            P[i++].set(v.x, v.y, v.z + voxExtZ);
                            P[i++].set(v.x, v.y, v.z);

                            P[i++].set(v.x, v.y + voxExtY, v.z);
                            P[i++].set(v.x + voxExtX, v.y + voxExtY, v.z);
                            P[i++].set(v.x + voxExtX, v.y + voxExtY, v.z);
                            P[i++].set(v.x + voxExtX, v.y + voxExtY, v.z + voxExtZ);
                            P[i++].set(v.x + voxExtX, v.y + voxExtY, v.z + voxExtZ);
                            P[i++].set(v.x, v.y + voxExtY, v.z + voxExtZ);
                            P[i++].set(v.x, v.y + voxExtY, v.z + voxExtZ);
                            P[i++].set(v.x, v.y + voxExtY, v.z);

                            P[i++].set(v.x, v.y, v.z);
                            P[i++].set(v.x, v.y + voxExtY, v.z);
                            P[i++].set(v.x + voxExtX, v.y, v.z);
                            P[i++].set(v.x + voxExtX, v.y + voxExtY, v.z);
                            P[i++].set(v.x + voxExtX, v.y, v.z + voxExtZ);
                            P[i++].set(v.x + voxExtX, v.y + voxExtY, v.z + voxExtZ);
                            P[i++].set(v.x, v.y, v.z + voxExtZ);
                            P[i++].set(v.x, v.y + voxExtY, v.z + voxExtZ);
                        }
                        curVoxel++;
                    }
                }
            }

            _bufP.generate(P, i, 3);
            delete[] P;
        }

        _bufP.drawArrayAsConstantColorLines(SLCol3f::CYAN);
    }
}
//-----------------------------------------------------------------------------
/*!
Ray Mesh intersection method using the regular grid space subdivision structure
and a voxel traversal algorithm described in "A Fast Voxel Traversal Algorithm
for Ray Tracing" by John Amanatides and Andrew Woo.
*/
SLbool SLCompactGrid::intersect (SLRay* ray, SLNode* node)
{
	// Check first if the AABB is hit at all
	if (node->aabb()->isHitInOS(ray))
	{
		SLbool wasHit = false;

		if (_voxelCnt > 0)
		{  //Calculate the intersection point with the AABB
			SLVec3f O = ray->originOS;
			SLVec3f D = ray->dirOS;
			SLVec3f invD = ray->invDirOS;
			SLVec3f startPoint = O;

            ////Determine start voxel of the grid
            if (ray->tmin > 0)  startPoint += ray->tmin*D;
            SLVec3i startVox = containingVoxel(startPoint);

			// Calculate the voxel ID into our 1D-voxel array
			SLuint voxID = startVox.x + startVox.y*_size.x + startVox.z*_size.x*_size.y;

			// Calculate steps: -1 or 1 on each axis
			SLint stepX = (D.x > 0) ? 1 : (D.x < 0) ? -1 : 0;
			SLint stepY = (D.y > 0) ? 1 : (D.y < 0) ? -1 : 0;
			SLint stepZ = (D.z > 0) ? 1 : (D.z < 0) ? -1 : 0;

			// Calculate the min. & max point of the start voxel
			SLVec3f minVox(_minV.x + startVox.x*_voxelWidth,
						   _minV.y + startVox.y*_voxelWidth,
						   _minV.z + startVox.z*_voxelWidth);
			SLVec3f maxVox(minVox.x + _voxelWidth,
				           minVox.y + _voxelWidth,
				           minVox.z + _voxelWidth);

			// Calculate max. dist along the ray for each component in tMaxX,Y,Z
			SLfloat tMaxX = FLT_MAX, tMaxY = FLT_MAX, tMaxZ = FLT_MAX;
			if (stepX ==  1) tMaxX = (maxVox.x - O.x) * invD.x; else
			if (stepX == -1) tMaxX = (minVox.x - O.x) * invD.x;
			if (stepY ==  1) tMaxY = (maxVox.y - O.y) * invD.y; else
			if (stepY == -1) tMaxY = (minVox.y - O.y) * invD.y;
			if (stepZ ==  1) tMaxZ = (maxVox.z - O.z) * invD.z; else
			if (stepZ == -1) tMaxZ = (minVox.z - O.z) * invD.z;

			// tMax is max. distance along the ray to stay in the current voxel
			SLfloat tMax = SL_min(tMaxX, tMaxY, tMaxZ);

			// Precalculate the voxel id increment
			SLint incIDX = stepX;
			SLint incIDY = stepY*_size.x;
			SLint incIDZ = stepZ*_size.x*_size.y;

			// Calculate tDeltaX,Y & Z (=dist. along the ray in a voxel)
            SLfloat tDeltaX = (_voxelWidth * invD.x) * stepX;
            SLfloat tDeltaY = (_voxelWidth * invD.y) * stepY;
            SLfloat tDeltaZ = (_voxelWidth * invD.z) * stepZ;

			// Now traverse the voxels
			while (!wasHit)
			{
                for (SLuint i = _voxelOffsets[voxID]; i < _voxelOffsets[voxID + 1]; ++i)
				{   if (_m->hitTriangleOS(ray, node, _triangleIndexes[i] * 3))
					{   if (ray->length <= tMax && !wasHit)
							wasHit = true;
					}
				}

				//step Voxel
				if (tMaxX < tMaxY)
				{
					if (tMaxX < tMaxZ)
					{
						startVox.x += stepX;
						if (startVox.x >= _size.x || startVox.x < 0) return wasHit; // dropped outside grid
						tMaxX += tDeltaX;
						voxID += incIDX;
						tMax = tMaxX;
					}
					else
					{
						startVox.z += stepZ;
						if (startVox.z >= _size.z || startVox.z < 0) return wasHit; // dropped outside grid
						tMaxZ += tDeltaZ;
						voxID += incIDZ;
						tMax = tMaxZ;
					}
				}
				else
				{
					if (tMaxY < tMaxZ)
					{
						startVox.y += stepY;
						if (startVox.y >= _size.y || startVox.y < 0) return wasHit; // dropped outside grid
						tMaxY += tDeltaY;
						voxID += incIDY;
						tMax = tMaxY;
					}
					else
					{
						startVox.z += stepZ;
						if (startVox.z >= _size.z || startVox.z < 0) return wasHit; // dropped outside grid
						tMaxZ += tDeltaZ;
						voxID += incIDZ;
						tMax = tMaxZ;
					}
				}
			}
			return wasHit;
		}
		else
		{  // not enough triangles for regular grid > check them all
			for (SLuint t = 0; t<_m->numI; t += 3)
			{
				if (_m->hitTriangleOS(ray, node, t) && !wasHit) wasHit = true;
			}
			return wasHit;
		}
	}
	else return false; // did not hit aabb
}
//-----------------------------------------------------------------------------
