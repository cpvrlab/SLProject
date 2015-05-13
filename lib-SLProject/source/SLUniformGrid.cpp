//#############################################################################
//  File:      SLUniformGrid.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

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
    _vox         = 0;
    _voxCnt      = 0;
    _voxCntEmpty = 0;
    _voxMaxTria  = 0;
    _voxAvgTria  = 0;
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
    if (_vox)
    {   for (SLuint i=0; i<_voxCnt; ++i) 
        {   if (_vox[i])
            {   _vox[i]->clear();
                delete _vox[i];
            }
        }
        delete[] _vox;
    }

    _vox         = 0;
    _voxCnt      = 0;
    _voxCntEmpty = 0;
    _voxMaxTria  = 0;
    _voxAvgTria  = 0;
    
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
    // Calc. voxel resolution, extent and allocate voxel array
    SLVec3f size = _maxV - _minV;
   
    // Woo's method
    SLfloat voxDensity = 20.0f;
    SLuint  numT = _m->numI / 3; // NO. of triangles
    SLfloat nr = (SLfloat)pow(voxDensity*numT,1.0f/3.0f);
    SLfloat maxS = SL_max(size.x, size.y, size.z);
    _voxResX = max(1, (SLint)(nr*size.x/maxS));
    _voxResY = max(1, (SLint)(nr*size.y/maxS));
    _voxResZ = max(1, (SLint)(nr*size.z/maxS));
   
    _voxResXY= _voxResX * _voxResY;
    _voxCnt  = _voxResZ * _voxResXY;
    _voxExtX = size.x / _voxResX;
    _voxExtY = size.y / _voxResY;
    _voxExtZ = size.z / _voxResZ;
   
    // Allocate array of pointer to SLV32ushort
    _vox = new SLV32ushort*[_voxCnt];
    for (SLuint i=0; i<_voxCnt; ++i) _vox[i] = 0;

    SLint    x, y, z;
    SLuint   i, curVoxel = 0;
    SLfloat  boxHalfExt[3] = {_voxExtX*0.5f, _voxExtY*0.5f, _voxExtZ*0.5f};
    SLfloat  curVoxelCenter[3];
    SLfloat  vert[3][3];
    SLuint   voxCntNotEmpty = 0;
    
    // Loop through all triangles and assign them to the voxels
    for(SLuint t = 0; t < numT; ++t)
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
        SLVec3f minT = SLVec3f(SL_min(vert[0][0], vert[1][0], vert[2][0]),
                               SL_min(vert[0][1], vert[1][1], vert[2][1]),
                               SL_min(vert[0][2], vert[1][2], vert[2][2]));
        SLVec3f maxT = SLVec3f(SL_max(vert[0][0], vert[1][0], vert[2][0]),
                               SL_max(vert[0][1], vert[1][1], vert[2][1]),
                               SL_max(vert[0][2], vert[1][2], vert[2][2]));
      
        // min voxel index of triangle
        SLint minx = (SLint)((minT.x-_minV.x) / _voxExtX);
        SLint miny = (SLint)((minT.y-_minV.y) / _voxExtY);
        SLint minz = (SLint)((minT.z-_minV.z) / _voxExtZ);
        // max voxel index of triangle
        SLint maxx = (SLint)((maxT.x-_minV.x) / _voxExtX);
        SLint maxy = (SLint)((maxT.y-_minV.y) / _voxExtY);
        SLint maxz = (SLint)((maxT.z-_minV.z) / _voxExtZ);
        if (maxx >= _voxResX) maxx=_voxResX-1;
        if (maxy >= _voxResY) maxy=_voxResY-1;
        if (maxz >= _voxResZ) maxz=_voxResZ-1;
                                                   
        // Loop through voxels
        curVoxelCenter[2]  = _minV.z + minz*_voxExtZ + boxHalfExt[2];
        for (z=minz; z<=maxz; ++z, curVoxelCenter[2] += _voxExtZ) 
        {  
            curVoxelCenter[1]  = _minV.y + miny*_voxExtY + boxHalfExt[1];
            for (y=miny; y<=maxy; ++y, curVoxelCenter[1] += _voxExtY) 
            {  
                curVoxelCenter[0]  = _minV.x + minx*_voxExtX + boxHalfExt[0];
                for (x=minx; x<=maxx; ++x, curVoxelCenter[0] += _voxExtX) 
                {  
                    curVoxel = x + y*_voxResX + z*_voxResXY;
               
                    //triangle-AABB overlap test by Thomas Möller
                    if (triBoxOverlap(curVoxelCenter, boxHalfExt, vert))
                    //trianlgesAABB-AABB overlap test is faster but not as precise
                    //if (triBoxBoxOverlap(curVoxelCenter, boxHalfExt, vert)) 
                    {  
                        {  if (_vox[curVoxel] == 0) 
                            {   voxCntNotEmpty++;
                                _vox[curVoxel] = new SLV32ushort;
                            }
                            _vox[curVoxel]->push_back(t);

                            if (_vox[curVoxel]->size() > _voxMaxTria)
                                _voxMaxTria = _vox[curVoxel]->size();
                        }
                    }
                }
            }
        }
    }
   
    _voxCntEmpty = _voxCnt - voxCntNotEmpty;
   
    // Reduce dynamic arrays to real size
    for (i=0; i<_voxCnt; ++i) 
    {   if (_vox[i])
            _vox[i]->reserve(_vox[i]->size());
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
    if (_voxCnt > 0 && _vox)
    {  
        SLuint   i=0;  // number of lines to draw
         
        if (!_bufP.id())
        {   SLint    x, y, z;
            SLuint   curVoxel = 0;
            SLVec3f  v;
            SLuint   numP = 12*2*_voxResX*_voxResY*_voxResZ;
            SLVec3f* P = new SLVec3f[numP]; 
                                 
            // Loop through voxels
            v.z  = _minV.z;
            for (z=0; z<_voxResZ; ++z, v.z += _voxExtZ) 
            {   v.y  = _minV.y;
                for (y=0; y<_voxResY; ++y, v.y += _voxExtY) 
                {   v.x = _minV.x;
                    for (x=0; x<_voxResX; ++x, v.x += _voxExtX) 
                    {  
                        if (_vox[curVoxel] && _vox[curVoxel]->size() > 0)
                        {  
                            P[i++].set(v.x,          v.y,          v.z         ); 
                            P[i++].set(v.x+_voxExtX, v.y,          v.z         );
                            P[i++].set(v.x+_voxExtX, v.y,          v.z         );
                            P[i++].set(v.x+_voxExtX, v.y,          v.z+_voxExtZ);
                            P[i++].set(v.x+_voxExtX, v.y,          v.z+_voxExtZ);
                            P[i++].set(v.x,          v.y,          v.z+_voxExtZ);
                            P[i++].set(v.x,          v.y,          v.z+_voxExtZ);
                            P[i++].set(v.x,          v.y,          v.z         );
                     
                            P[i++].set(v.x,          v.y+_voxExtY, v.z         ); 
                            P[i++].set(v.x+_voxExtX, v.y+_voxExtY, v.z         );
                            P[i++].set(v.x+_voxExtX, v.y+_voxExtY, v.z         );
                            P[i++].set(v.x+_voxExtX, v.y+_voxExtY, v.z+_voxExtZ);
                            P[i++].set(v.x+_voxExtX, v.y+_voxExtY, v.z+_voxExtZ);
                            P[i++].set(v.x,          v.y+_voxExtY, v.z+_voxExtZ);
                            P[i++].set(v.x,          v.y+_voxExtY, v.z+_voxExtZ);
                            P[i++].set(v.x,          v.y+_voxExtY, v.z         ); 
                     
                            P[i++].set(v.x,          v.y,          v.z         ); 
                            P[i++].set(v.x,          v.y+_voxExtY, v.z         ); 
                            P[i++].set(v.x+_voxExtX, v.y         , v.z         );
                            P[i++].set(v.x+_voxExtX, v.y+_voxExtY, v.z         );
                            P[i++].set(v.x+_voxExtX, v.y         , v.z+_voxExtZ);
                            P[i++].set(v.x+_voxExtX, v.y+_voxExtY, v.z+_voxExtZ);
                            P[i++].set(v.x         , v.y         , v.z+_voxExtZ);
                            P[i++].set(v.x         , v.y+_voxExtY, v.z+_voxExtZ);
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
    stats.numBytesAccel += _voxCnt * sizeOfSLV32ushortP;   // _vox
    stats.numBytesAccel += (_voxCnt-_voxCntEmpty) * sizeOfSLV32ushort;   // _vox
    
    for (SLuint i=0; i<_voxCnt; ++i) // _vox[i].capacity
        if (_vox[i])
            stats.numBytesAccel += _vox[i]->capacity()*sizeof(SLuint);

    stats.numVoxels += _voxCnt;
    stats.numVoxEmpty += _voxCntEmpty;
    if (_voxMaxTria > stats.numVoxMaxTria)
        stats.numVoxMaxTria = _voxMaxTria;
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
      
        if (_voxCnt > 0)
        {  //Calculate the intersection point with the AABB
            SLVec3f O = ray->originOS;
            SLVec3f D = ray->dirOS;
            SLVec3f invD = ray->invDirOS;
            SLVec3f startPoint = O;
         
            if (ray->tmin > 0)  startPoint += ray->tmin*D;
         
            //Determine start voxel of the grid
            startPoint -= _minV;
            SLint x = (SLint)(startPoint.x / _voxExtX); // voxel index in x-dir
            SLint y = (SLint)(startPoint.y / _voxExtY); // voxel index in y-dir
            SLint z = (SLint)(startPoint.z / _voxExtZ); // voxel index in z-dir
         
            // Check bounds of voxel indexis
            if (x >= _voxResX) x=_voxResX-1; if (x < 0) x=0;
            if (y >= _voxResY) y=_voxResY-1; if (y < 0) y=0;
            if (z >= _voxResZ) z=_voxResZ-1; if (z < 0) z=0;
         
            // Calculate the voxel ID into our 1D-voxel array
            SLuint voxID = x + y*_voxResX + z*_voxResXY;
         
            // Calculate steps: -1 or 1 on each axis
            SLint stepX = (D.x > 0) ? 1 : (D.x < 0) ? -1 : 0;
            SLint stepY = (D.y > 0) ? 1 : (D.y < 0) ? -1 : 0;
            SLint stepZ = (D.z > 0) ? 1 : (D.z < 0) ? -1 : 0;

            // Calculate the min. & max point of the start voxel
            SLVec3f minVox(_minV.x + x*_voxExtX,
                           _minV.y + y*_voxExtY,
                           _minV.z + z*_voxExtZ);
            SLVec3f maxVox(minVox.x + _voxExtX,
                           minVox.y + _voxExtY,
                           minVox.z + _voxExtZ);
                        
            // Calculate max. dist along the ray for each component in tMaxX,Y,Z
            SLfloat tMaxX=FLT_MAX, tMaxY=FLT_MAX, tMaxZ=FLT_MAX;
            if (stepX== 1) tMaxX = (maxVox.x - O.x) * invD.x; else 
            if (stepX==-1) tMaxX = (minVox.x - O.x) * invD.x;
            if (stepY== 1) tMaxY = (maxVox.y - O.y) * invD.y; else 
            if (stepY==-1) tMaxY = (minVox.y - O.y) * invD.y;
            if (stepZ== 1) tMaxZ = (maxVox.z - O.z) * invD.z; else 
            if (stepZ==-1) tMaxZ = (minVox.z - O.z) * invD.z;
         
            // tMax is max. distance along the ray to stay in the current voxel
            SLfloat tMax = SL_min(tMaxX, tMaxY, tMaxZ);
         
            // Precalculate the voxel id increment
            SLint incIDX = stepX;
            SLint incIDY = stepY*_voxResX;
            SLint incIDZ = stepZ*_voxResXY;
         
            // Calculate tDeltaX,Y & Z (=dist. along the ray in a voxel)
            SLfloat tDeltaX = (_voxExtX * invD.x) * stepX;
            SLfloat tDeltaY = (_voxExtY * invD.y) * stepY;
            SLfloat tDeltaZ = (_voxExtZ * invD.z) * stepZ;
         
            // Now traverse the voxels
            while (!wasHit)
            {
                // intersect all triangle in current voxel
                if (_vox[voxID])
                {  
                    for(SLint t=0; t<_vox[voxID]->size(); ++t)
                    {  
                        SLV32ushort* v = _vox[voxID];

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
                        if(x >= _voxResX || x < 0) return wasHit; // dropped outside grid
                        tMaxX += tDeltaX;
                        voxID += incIDX;
                        tMax = tMaxX;
                    } else
                    {   z += stepZ;
                        if(z >= _voxResZ || z < 0) return wasHit; // dropped outside grid
                        tMaxZ += tDeltaZ;
                        voxID += incIDZ;
                        tMax = tMaxZ;
                    }
                } else
                {   if(tMaxY < tMaxZ)
                    {   y += stepY;
                        if(y >= _voxResY || y < 0) return wasHit; // dropped outside grid
                        tMaxY += tDeltaY;
                        voxID += incIDY;
                        tMax = tMaxY;
                    } else
                    {   z += stepZ;
                        if(z >= _voxResZ || z < 0) return wasHit; // dropped outside grid
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
