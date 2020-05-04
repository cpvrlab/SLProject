//#############################################################################
//  File:      SLUniformGrid.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers

#include <SLUniformGrid.h>
#include <SLNode.h>       
#include <SLRay.h>
#include <SLRaytracer.h>
#include <SLSceneView.h>
#include <SLCamera.h>
#include <SLGLProgram.h>
#include <TriangleBoxIntersect.h>
      
//-----------------------------------------------------------------------------
SLUniformGrid::SLUniformGrid(SLMesh* m) : SLAccelStruct(m)
{  
    _voxel         = 0;
    _voxelCnt      = 0;
    _voxelCntEmpty = 0;
    _voxelMaxTria  = 0;
    _voxelAvgTria  = 0;
}
//-----------------------------------------------------------------------------
SLUniformGrid::~SLUniformGrid()
{  
    deleteAll();
}
//-----------------------------------------------------------------------------
//! Deletes the entire uniform grid data
void SLUniformGrid::deleteAll()
{
    if (_voxel)
    {   for (SLuint i=0; i<_voxelCnt; ++i) 
        {   if (_voxel[i])
            {   _voxel[i]->clear();
                delete _voxel[i];
            }
        }
        delete[] _voxel;
    }

    _voxel         = 0;
    _voxelCnt      = 0;
    _voxelCntEmpty = 0;
    _voxelMaxTria  = 0;
    _voxelAvgTria  = 0;
    
    disposeBuffers();
}
//-----------------------------------------------------------------------------
/*! Builds the uniform grid for ray tracing acceleration
*/
void SLUniformGrid::build(SLVec3f minV, SLVec3f maxV)
{  
    _minV = minV;
    _maxV = maxV;
   
    deleteAll();
   
    // Calculate uniform grid 
    // Calculate voxel resolution, extent and allocate voxel array
    SLVec3f size = _maxV - _minV;
   
    // Woo's method
    const SLfloat DENSITY = 20.0f;
    SLuint  numTriangles = _m->numI / 3; // NO. of triangles
    SLfloat f = (SLfloat)pow(DENSITY*numTriangles,1.0f/3.0f);
    SLfloat maxS = std::max(size.x, size.y, size.z);
    _size.x = max(1, (SLint)(f*size.x/maxS));
    _size.y = max(1, (SLint)(f*size.y/maxS));
    _size.z = max(1, (SLint)(f*size.z/maxS));
    _voxelCnt = _size.x * _size.y * _size.z;
    _voxelSize.x = size.x / _size.x;
    _voxelSize.y = size.y / _size.y;
    _voxelSize.z = size.z / _size.z;
   
    // Allocate array of pointer to SLV32ushort
    _voxel = new SLV32ushort*[_voxelCnt];
    for (SLuint i=0; i<_voxelCnt; ++i) _voxel[i] = 0;

    SLint    x, y, z;
    SLuint   i, voxelID = 0;
    SLfloat  boxHalfExt[3] = {_voxelSize.x*0.5f, _voxelSize.y*0.5f, _voxelSize.z*0.5f};
    SLfloat  curVoxelCenter[3];
    SLfloat  vert[3][3];
    SLuint   voxCntNotEmpty = 0;
    
    // Loop through all triangles and assign them to the voxels
    for(SLuint t = 0; t < numTriangles; ++t)
    {  
        // Copy triangle vertices into SLfloat array[3][3]
        SLuint i = t * 3;
        if (_m->I16)
        {   vert[0][0] = _m->finalP()[_m->I16[i  ]].x;
            vert[0][1] = _m->finalP()[_m->I16[i  ]].y;
            vert[0][2] = _m->finalP()[_m->I16[i  ]].z;
            vert[1][0] = _m->finalP()[_m->I16[i+1]].x;
            vert[1][1] = _m->finalP()[_m->I16[i+1]].y;
            vert[1][2] = _m->finalP()[_m->I16[i+1]].z;
            vert[2][0] = _m->finalP()[_m->I16[i+2]].x;
            vert[2][1] = _m->finalP()[_m->I16[i+2]].y;
            vert[2][2] = _m->finalP()[_m->I16[i+2]].z;
        } else
        {   vert[0][0] = _m->finalP()[_m->I32[i  ]].x;
            vert[0][1] = _m->finalP()[_m->I32[i  ]].y;
            vert[0][2] = _m->finalP()[_m->I32[i  ]].z;
            vert[1][0] = _m->finalP()[_m->I32[i+1]].x;
            vert[1][1] = _m->finalP()[_m->I32[i+1]].y;
            vert[1][2] = _m->finalP()[_m->I32[i+1]].z;
            vert[2][0] = _m->finalP()[_m->I32[i+2]].x;
            vert[2][1] = _m->finalP()[_m->I32[i+2]].y;
            vert[2][2] = _m->finalP()[_m->I32[i+2]].z;
        }
        // Min. and max. point of triangle
        SLVec3f minT = SLVec3f(std::min(vert[0][0], vert[1][0], vert[2][0]),
                               std::min(vert[0][1], vert[1][1], vert[2][1]),
                               std::min(vert[0][2], vert[1][2], vert[2][2]));
        SLVec3f maxT = SLVec3f(std::max(vert[0][0], vert[1][0], vert[2][0]),
                               std::max(vert[0][1], vert[1][1], vert[2][1]),
                               std::max(vert[0][2], vert[1][2], vert[2][2]));
      
        // min voxel index of triangle
        SLint minx = (SLint)((minT.x-_minV.x) / _voxelSize.x);
        SLint miny = (SLint)((minT.y-_minV.y) / _voxelSize.y);
        SLint minz = (SLint)((minT.z-_minV.z) / _voxelSize.z);
        // max voxel index of triangle
        SLint maxx = (SLint)((maxT.x-_minV.x) / _voxelSize.x);
        SLint maxy = (SLint)((maxT.y-_minV.y) / _voxelSize.y);
        SLint maxz = (SLint)((maxT.z-_minV.z) / _voxelSize.z);
        if (maxx >= _size.x) maxx=_size.x-1;
        if (maxy >= _size.y) maxy=_size.y-1;
        if (maxz >= _size.z) maxz=_size.z-1;
                                                   
        // Loop through voxels
        curVoxelCenter[2]  = _minV.z + minz*_voxelSize.z + boxHalfExt[2];
        for (z=minz; z<=maxz; ++z, curVoxelCenter[2] += _voxelSize.z) 
        {  
            curVoxelCenter[1]  = _minV.y + miny*_voxelSize.y + boxHalfExt[1];
            for (y=miny; y<=maxy; ++y, curVoxelCenter[1] += _voxelSize.y) 
            {  
                curVoxelCenter[0]  = _minV.x + minx*_voxelSize.x + boxHalfExt[0];
                for (x=minx; x<=maxx; ++x, curVoxelCenter[0] += _voxelSize.x) 
                {  
                    voxelID = x + y*_size.x + z*_size.x*_size.y;
               
                    //triangle-AABB overlap test by Thomas M�ller
                    if (triBoxOverlap(curVoxelCenter, boxHalfExt, vert))
                    //trianlgesAABB-AABB overlap test is faster but not as precise
                    //if (triBoxBoxOverlap(curVoxelCenter, boxHalfExt, vert)) 
                    {  
                        {   if (_voxel[voxelID] == 0)
                            {   voxCntNotEmpty++;
                                _voxel[voxelID] = new SLV32ushort;
                            }
                            _voxel[voxelID]->push_back(t);

                            if (_voxel[voxelID]->size() > _voxelMaxTria)
                                _voxelMaxTria = _voxel[voxelID]->size();
                        }
                    }
                }
            }
        }
    }
   
    _voxelCntEmpty = _voxelCnt - voxCntNotEmpty;
   
    // Reduce dynamic arrays to real size
    for (i=0; i<_voxelCnt; ++i) 
    {   if (_voxel[i])
            _voxel[i]->reserve(_voxel[i]->size());
    }

    /*
    // dump for debugging
    SL_LOG("\nMesh name: %s\n", _m->name().c_str());
    SL_LOG("Number of voxels: %d\n", _voxCnt);
    SL_LOG("Empty voxels: %4.0f%%\n", 
            (SLfloat)(_voxCnt-voxCntNotEmpty)/(SLfloat)_voxCnt * 100.0f);
    SL_LOG("Avg. tria. per voxel: %4.1f\n", (SLfloat)_m->numF/(SLfloat)voxCntNotEmpty);
             
    // dump voxels
    curVoxel = 0;
    curVoxelCenter[2]  = _minV.z + boxHalfExt[2];
    for (z=0; z<_voxResZ; ++z, curVoxelCenter[2] += _voxExtZ) 
    {  
        curVoxelCenter[1]  = _minV.y + boxHalfExt[1];
        for (y=0; y<_voxResY; ++y, curVoxelCenter[1] += _voxExtY) 
        {  
            curVoxelCenter[0]  = _minV.x + boxHalfExt[0];
            for (x=0; x<_voxResX; ++x, curVoxelCenter[0] += _voxExtX) 
            {              
            SL_LOG("\t%0d(%3.1f,%3.1f,%3.1f):%0d " ,curVoxel, 
                    curVoxelCenter[0], curVoxelCenter[1], curVoxelCenter[2],
                    _vox[curVoxel].size());
            curVoxel++;
            }
            SL_LOG("\n");
        }
        SL_LOG("\n");
    }
    */
}

//-----------------------------------------------------------------------------
/*! draws the the uniform grid voxels.
*/
void SLUniformGrid::draw(SLSceneView* sv) 
{        
    // Draw regular grid
    if (_voxelCnt > 0 && _voxel)
    {  
        SLuint   i=0;  // number of lines to draw
         
        if (!_bufP.id())
        {   SLint    x, y, z;
            SLuint   curVoxel = 0;
            SLVec3f  v;
            SLuint   numP = 12*2*_size.x*_size.y*_size.z;
            SLVec3f* P = new SLVec3f[numP]; 
                                 
            // Loop through voxels
            v.z  = _minV.z;
            for (z=0; z<_size.z; ++z, v.z += _voxelSize.z) 
            {   v.y  = _minV.y;
                for (y=0; y<_size.y; ++y, v.y += _voxelSize.y) 
                {   v.x = _minV.x;
                    for (x=0; x<_size.x; ++x, v.x += _voxelSize.x) 
                    {  
                        if (_voxel[curVoxel] && !_voxel[curVoxel]->empty())
                        {  
                            P[i++].set(v.x,          v.y,          v.z         ); 
                            P[i++].set(v.x+_voxelSize.x, v.y,          v.z         );
                            P[i++].set(v.x+_voxelSize.x, v.y,          v.z         );
                            P[i++].set(v.x+_voxelSize.x, v.y,          v.z+_voxelSize.z);
                            P[i++].set(v.x+_voxelSize.x, v.y,          v.z+_voxelSize.z);
                            P[i++].set(v.x,          v.y,          v.z+_voxelSize.z);
                            P[i++].set(v.x,          v.y,          v.z+_voxelSize.z);
                            P[i++].set(v.x,          v.y,          v.z         );
                     
                            P[i++].set(v.x,          v.y+_voxelSize.y, v.z         ); 
                            P[i++].set(v.x+_voxelSize.x, v.y+_voxelSize.y, v.z         );
                            P[i++].set(v.x+_voxelSize.x, v.y+_voxelSize.y, v.z         );
                            P[i++].set(v.x+_voxelSize.x, v.y+_voxelSize.y, v.z+_voxelSize.z);
                            P[i++].set(v.x+_voxelSize.x, v.y+_voxelSize.y, v.z+_voxelSize.z);
                            P[i++].set(v.x,          v.y+_voxelSize.y, v.z+_voxelSize.z);
                            P[i++].set(v.x,          v.y+_voxelSize.y, v.z+_voxelSize.z);
                            P[i++].set(v.x,          v.y+_voxelSize.y, v.z         ); 
                     
                            P[i++].set(v.x,          v.y,          v.z         ); 
                            P[i++].set(v.x,          v.y+_voxelSize.y, v.z         ); 
                            P[i++].set(v.x+_voxelSize.x, v.y         , v.z         );
                            P[i++].set(v.x+_voxelSize.x, v.y+_voxelSize.y, v.z         );
                            P[i++].set(v.x+_voxelSize.x, v.y         , v.z+_voxelSize.z);
                            P[i++].set(v.x+_voxelSize.x, v.y+_voxelSize.y, v.z+_voxelSize.z);
                            P[i++].set(v.x         , v.y         , v.z+_voxelSize.z);
                            P[i++].set(v.x         , v.y+_voxelSize.y, v.z+_voxelSize.z);
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
/*! Updates the parent groups statistics.
*/
void SLUniformGrid::updateStats(SLNodeStats &stats)
{   
    SLint sizeOfSLUniformGrid = sizeof(SLUniformGrid);
    SLint sizeOfSLV32ushort   = sizeof(SLV32ushort);
    SLint sizeOfSLV32ushortP  = sizeof(SLV32ushort*);
    stats.numBytesAccel += sizeOfSLUniformGrid;
    stats.numBytesAccel += _voxelCnt * sizeOfSLV32ushortP;   // _vox
    stats.numBytesAccel += (_voxelCnt-_voxelCntEmpty) * sizeOfSLV32ushort;   // _vox
    
    for (SLuint i=0; i<_voxelCnt; ++i) // _vox[i].capacity
        if (_voxel[i])
            stats.numBytesAccel += _voxel[i]->capacity()*sizeof(SLuint);

    stats.numVoxels += _voxelCnt;
    stats.numVoxEmpty += _voxelCntEmpty;
    if (_voxelMaxTria > stats.numVoxMaxTria)
        stats.numVoxMaxTria = _voxelMaxTria;
}

//-----------------------------------------------------------------------------
/*!
Ray Mesh intersection method using the regular grid space subdivision structure
and a voxel traversal algorithm described in "A Fast Voxel Traversal Algorithm
for Ray Tracing" by John Amanatides and Andrew Woo.
*/
SLbool SLUniformGrid::intersect(SLRay* ray, SLNode* node)
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
         
            if (ray->tmin > 0)  startPoint += ray->tmin*D;
         
            //Determine start voxel of the grid
            startPoint -= _minV;
            SLint x = (SLint)(startPoint.x / _voxelSize.x); // voxel index in x-dir
            SLint y = (SLint)(startPoint.y / _voxelSize.y); // voxel index in y-dir
            SLint z = (SLint)(startPoint.z / _voxelSize.z); // voxel index in z-dir
         
            // Check bounds of voxel indexes
            if (x >= _size.x) x=_size.x-1; if (x < 0) x=0;
            if (y >= _size.y) y=_size.y-1; if (y < 0) y=0;
            if (z >= _size.z) z=_size.z-1; if (z < 0) z=0;
         
            // Calculate the voxel ID into our 1D-voxel array
            SLuint voxID = x + y*_size.x + z*_size.x*_size.y;
         
            // Calculate steps: -1 or 1 on each axis
            SLint stepX = (D.x > 0) ? 1 : (D.x < 0) ? -1 : 0;
            SLint stepY = (D.y > 0) ? 1 : (D.y < 0) ? -1 : 0;
            SLint stepZ = (D.z > 0) ? 1 : (D.z < 0) ? -1 : 0;

            // Calculate the min. & max point of the start voxel
            SLVec3f minVox(_minV.x + x*_voxelSize.x,
                           _minV.y + y*_voxelSize.y,
                           _minV.z + z*_voxelSize.z);
            SLVec3f maxVox(minVox.x + _voxelSize.x,
                           minVox.y + _voxelSize.y,
                           minVox.z + _voxelSize.z);
                        
            // Calculate max. dist along the ray for each component in tMaxX,Y,Z
            SLfloat tMaxX=FLT_MAX, tMaxY=FLT_MAX, tMaxZ=FLT_MAX;
            if (stepX== 1) tMaxX = (maxVox.x - O.x) * invD.x; else 
            if (stepX==-1) tMaxX = (minVox.x - O.x) * invD.x;
            if (stepY== 1) tMaxY = (maxVox.y - O.y) * invD.y; else 
            if (stepY==-1) tMaxY = (minVox.y - O.y) * invD.y;
            if (stepZ== 1) tMaxZ = (maxVox.z - O.z) * invD.z; else 
            if (stepZ==-1) tMaxZ = (minVox.z - O.z) * invD.z;
         
            // tMax is max. distance along the ray to stay in the current voxel
            SLfloat tMax = std::min(tMaxX, tMaxY, tMaxZ);
         
            // Precalculate the voxel id increment
            SLint incIDX = stepX;
            SLint incIDY = stepY*_size.x;
            SLint incIDZ = stepZ*_size.x*_size.y;
         
            // Calculate tDeltaX,Y & Z (=dist. along the ray in a voxel)
            SLfloat tDeltaX = (_voxelSize.x * invD.x) * stepX;
            SLfloat tDeltaY = (_voxelSize.y * invD.y) * stepY;
            SLfloat tDeltaZ = (_voxelSize.z * invD.z) * stepZ;
         
            // Now traverse the voxels
            while (!wasHit)
            {
                // intersect all triangle in current voxel
                if (_voxel[voxID])
                {  
                    for(SLint t=0; t<_voxel[voxID]->size(); ++t)
                    {  
                        SLV32ushort* v = _voxel[voxID];

                        if (_m->hitTriangleOS(ray, node, v->at(t)*3))
                        {  
                            // test whether intersection point inside current voxel
                            if (ray->length <= tMax && !wasHit) 
                                wasHit = true;
                        }
                    }
                }

                //step Voxel
                if(tMaxX < tMaxY)
                {  if(tMaxX < tMaxZ)
                    {   x += stepX;
                        if(x >= _size.x || x < 0) return wasHit; // dropped outside grid
                        tMaxX += tDeltaX;
                        voxID += incIDX;
                        tMax = tMaxX;
                    } else
                    {   z += stepZ;
                        if(z >= _size.z || z < 0) return wasHit; // dropped outside grid
                        tMaxZ += tDeltaZ;
                        voxID += incIDZ;
                        tMax = tMaxZ;
                    }
                } else
                {   if(tMaxY < tMaxZ)
                    {   y += stepY;
                        if(y >= _size.y || y < 0) return wasHit; // dropped outside grid
                        tMaxY += tDeltaY;
                        voxID += incIDY;
                        tMax = tMaxY;
                    } else
                    {   z += stepZ;
                        if(z >= _size.z || z < 0) return wasHit; // dropped outside grid
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
            for (SLuint t=0; t<_m->numI; t+=3)
            {   if(_m->hitTriangleOS(ray, node, t) && !wasHit) wasHit = true;
            }
            return wasHit;
        }
    } 
    else return false; // did not hit aabb
}

//-----------------------------------------------------------------------------
