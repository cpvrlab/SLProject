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
SLCompactGrid::SLCompactGrid(SLMesh* m) : 
    SLAccelStruct(m), _numVoxels(0), _emptyVoxels(0), _maxTrianglesPerVoxel(0)
{}
//-----------------------------------------------------------------------------
SLVec3i SLCompactGrid::containingCell(const SLVec3f &p) const
{
    SLVec3i pos;
    SLVec3f delta = p - _minV;
    pos.x = (SLint)(delta.x / _cellWidth);
    pos.y = (SLint)(delta.y / _cellWidth);
    pos.z = (SLint)(delta.z / _cellWidth);

	if (pos.x >= _size.x || pos.y >= _size.y || pos.z >= _size.z)
		int i = 0;

    return pos;
}
//-----------------------------------------------------------------------------
int triBoxOverlap(SLVec3f center,
                  SLVec3f halfSize,
                  std::array<SLVec3f,3> triangle)
{
    return triBoxOverlap(*((float(*)[3])&center),
                         *((float(*)[3])&halfSize),
                         *((float(*)[3][3])&triangle));
}
//-----------------------------------------------------------------------------
void SLCompactGrid::setMinMaxCell(const Triangle &triangle,
                                  SLVec3i &minCell,
                                  SLVec3i &maxCell)
{
    minCell = maxCell = containingCell(triangle[0]);
    for (int i = 1; i < 3; ++i)
    {   auto &vertex = triangle[i];
        minCell.setMin(containingCell(vertex));
        maxCell.setMax(containingCell(vertex));
    }
}
//-----------------------------------------------------------------------------
void SLCompactGrid::build (SLVec3f minV, SLVec3f maxV)
{
    static const float density = 8;
    int numTriangles = _m->numI / 3;

    _minV = minV;
    _maxV = maxV;

    // Calculate grid size
    SLVec3f diagonal = _maxV - _minV;
    SLfloat volume = diagonal.x * diagonal.y * diagonal.z;
    float f = cbrtf(density*numTriangles / volume);
    _cellWidth = SL_min(SL_min(diagonal.x / ceil(diagonal.x*f),
                               diagonal.y / ceil(diagonal.y*f)),
                               diagonal.z / ceil(diagonal.z*f));
    _size.x = (SLint)ceil(diagonal.x / _cellWidth);
    _size.y = (SLint)ceil(diagonal.y / _cellWidth);
    _size.z = (SLint)ceil(diagonal.z / _cellWidth);

    SLVec3f boxHalfSize(_cellWidth / 2, _cellWidth / 2, _cellWidth / 2);

    _numVoxels = _size.x * _size.y * _size.z;
    _cellOffsets.assign(_numVoxels+1, 0);


    for (int i = 0; i < numTriangles; ++i)
    {
        auto index  = [&](int j) { return _m->I16 ? _m->I16[i * 3 + j] : _m->I32[i * 3 + j]; };
        auto vertex = [&](int j) { return _m->finalP()[index(j)]; };
        Triangle triangle = { vertex(0), vertex(1), vertex(2) };
        SLVec3i min, max, pos;
        setMinMaxCell(triangle, min, max);

        for (pos.x = min.x; pos.x <= max.x; ++pos.x)
        {   for (pos.y = min.y; pos.y <= max.y; ++pos.y)
            {   for (pos.z = min.z; pos.z <= max.z; ++pos.z)
                {   
                    int cellIndex = indexAtPos(pos);
                    SLVec3f cellCenter = _minV + SLVec3f((pos.x + .5f)*_cellWidth,
                                                         (pos.y + .5f)*_cellWidth,
                                                         (pos.z + .5f)*_cellWidth);
                    if (triBoxOverlap(cellCenter, boxHalfSize, triangle))
                        //++_cellOffsets[cellIndex].count;
                        ++_cellOffsets[cellIndex];
                }
            }
        }
    }

    //The last counter doesn't count and is always empty.
    //_maxTrianglesPerVoxel = _cellOffsets[0].count;
    //_emptyVoxels = (_cellOffsets[0].count == 0) - 1;
    //for (int i = 1; i < _cellOffsets.size(); ++i)
    //{
    //    _maxTrianglesPerVoxel = SL_max(_maxTrianglesPerVoxel, _cellOffsets[i].count.load());
    //    _emptyVoxels += _cellOffsets[i].count == 0;
    //    _cellOffsets[i].count += _cellOffsets[i-1].count;
    //}

    //_triangleIndices.resize(_cellOffsets.back().count);

    _maxTrianglesPerVoxel = _cellOffsets[0];
    _emptyVoxels = (_cellOffsets[0] == 0) - 1;
    for (int i = 1; i < _cellOffsets.size(); ++i)
    {
        _maxTrianglesPerVoxel = SL_max(_maxTrianglesPerVoxel, (SLuint)_cellOffsets[i]);
        _emptyVoxels += _cellOffsets[i] == 0;
        _cellOffsets[i] += _cellOffsets[i - 1];
    }

    _triangleIndices.resize(_cellOffsets.back());


    // Reverse iterate over triangles
    //#pragma omp parallel for default(shared)
    for (int i = numTriangles - 1; i >= 0; --i)
    {
        auto index  = [&](int j) { return _m->I16 ? _m->I16[i * 3 + j] : _m->I32[i * 3 + j]; };
        auto vertex = [&](int j) { return _m->finalP()[index(j)]; };
        Triangle triangle = { vertex(0), vertex(1), vertex(2) };
        SLVec3i min, max, pos;
        setMinMaxCell(triangle, min, max);

        for (pos.x = min.x; pos.x <= max.x; ++pos.x)
        {   for (pos.y = min.y; pos.y <= max.y; ++pos.y)
            {   for (pos.z = min.z; pos.z <= max.z; ++pos.z)
                {   
                    int cellIndex = indexAtPos(pos);
                    SLVec3f cellCenter = _minV + SLVec3f((pos.x + .5f)*_cellWidth,
                                                         (pos.y + .5f)*_cellWidth,
                                                         (pos.z + .5f)*_cellWidth);
                    if (triBoxOverlap(cellCenter, boxHalfSize, triangle))
                    {   int location = --_cellOffsets[cellIndex];
                        _triangleIndices[location] = i;
                    }
                }
            }
        }
    }

    _cellOffsets.shrink_to_fit();
    _triangleIndices.shrink_to_fit();
}
//-----------------------------------------------------------------------------
template<class T>
inline int objectSize(const T &vector)
{
    return sizeof(T);
}
//-----------------------------------------------------------------------------
template<class T>
inline SLint vectorSize(const T &vector)
{
    return (SLint)(vector.capacity()*sizeof(typename T::value_type));
}
//-----------------------------------------------------------------------------
void SLCompactGrid::updateStats (SLNodeStats &stats)
{
    stats.numVoxels += _numVoxels;
    stats.numVoxEmpty += _emptyVoxels;

    stats.numBytesAccel += objectSize(*this);
    stats.numBytesAccel += vectorSize(_cellOffsets);
    stats.numBytesAccel += vectorSize(_triangleIndices);

    stats.numVoxMaxTria = SL_max(_maxTrianglesPerVoxel, stats.numVoxMaxTria);
}
//-----------------------------------------------------------------------------
void SLCompactGrid::draw (SLSceneView* sv)
{
	return;
}
//-----------------------------------------------------------------------------
SLbool SLCompactGrid::intersect (SLRay* ray, SLNode* node)
{
	// Check first if the AABB is hit at all
	if (node->aabb()->isHitInOS(ray))
	{
		SLbool wasHit = false;

		if (_numVoxels > 0)
		{  //Calculate the intersection point with the AABB
			SLVec3f O = ray->originOS;
			SLVec3f D = ray->dirOS;
			SLVec3f invD = ray->invDirOS;
			SLVec3f startPoint = O;

			if (ray->tmin > 0)  startPoint += ray->tmin*D;

			//Determine start voxel of the grid
			startPoint -= _minV;
			SLint x = (SLint)(startPoint.x / _cellWidth); // voxel index in x-dir
			SLint y = (SLint)(startPoint.y / _cellWidth); // voxel index in y-dir
			SLint z = (SLint)(startPoint.z / _cellWidth); // voxel index in z-dir

			// Check bounds of voxel indexes
			if (x >= _size.x) x = _size.x - 1; if (x < 0) x = 0;
			if (y >= _size.y) y = _size.y - 1; if (y < 0) y = 0;
			if (z >= _size.z) z = _size.z - 1; if (z < 0) z = 0;

			// Calculate the voxel ID into our 1D-voxel array
			SLuint voxID = x + y*_size.x + z*_size.x*_size.y;

			// Calculate steps: -1 or 1 on each axis
			SLint stepX = (D.x > 0) ? 1 : (D.x < 0) ? -1 : 0;
			SLint stepY = (D.y > 0) ? 1 : (D.y < 0) ? -1 : 0;
			SLint stepZ = (D.z > 0) ? 1 : (D.z < 0) ? -1 : 0;

			// Calculate the min. & max point of the start voxel
			SLVec3f minVox(_minV.x + x*_cellWidth,
						   _minV.y + y*_cellWidth,
						   _minV.z + z*_cellWidth);
			SLVec3f maxVox(minVox.x + _cellWidth,
				           minVox.y + _cellWidth,
				           minVox.z + _cellWidth);

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
            SLfloat tDeltaX = (_cellWidth * invD.x) * stepX;
            SLfloat tDeltaY = (_cellWidth * invD.y) * stepY;
            SLfloat tDeltaZ = (_cellWidth * invD.z) * stepZ;

			// Now traverse the voxels
			while (!wasHit)
			{
				typedef std::remove_reference<decltype(_triangleIndices.front())>::type TriangleIndex;
				struct Range
				{
					TriangleIndex *begin() const { return a;  }
					TriangleIndex *end() const { return b;  }

					size_t size() { return b - a; }

					TriangleIndex *a;
					TriangleIndex *b;
				};

				Range triangles{ _triangleIndices.data() + _cellOffsets[voxID],
					             _triangleIndices.data() + _cellOffsets[voxID + 1] };

				if (triangles.size() > 0)
				for (auto &index : triangles)
				{
					if (_m->hitTriangleOS(ray, node, index * 3))
					{
						if (ray->length <= tMax && !wasHit)
							wasHit = true;
					}
				}

				//step Voxel
				if (tMaxX < tMaxY)
				{
					if (tMaxX < tMaxZ)
					{
						x += stepX;
						if (x >= _size.x || x < 0) return wasHit; // dropped outside grid
						tMaxX += tDeltaX;
						voxID += incIDX;
						tMax = tMaxX;
					}
					else
					{
						z += stepZ;
						if (z >= _size.z || z < 0) return wasHit; // dropped outside grid
						tMaxZ += tDeltaZ;
						voxID += incIDZ;
						tMax = tMaxZ;
					}
				}
				else
				{
					if (tMaxY < tMaxZ)
					{
						y += stepY;
						if (y >= _size.y || y < 0) return wasHit; // dropped outside grid
						tMaxY += tDeltaY;
						voxID += incIDY;
						tMax = tMaxY;
					}
					else
					{
						z += stepZ;
						if (z >= _size.z || z < 0) return wasHit; // dropped outside grid
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
