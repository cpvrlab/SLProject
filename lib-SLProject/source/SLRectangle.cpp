//#############################################################################
//  File:      SLRectangle.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLRectangle.h>

//-----------------------------------------------------------------------------
//! SLRectangle ctor with min & max corners and its resolutions 
SLRectangle::SLRectangle(SLVec2f min, SLVec2f max,
                         SLuint resX, SLuint resY, 
                         SLstring name,
                         SLMaterial* mat) : SLMesh(name) 
{
    assert(min!=max);
    assert(resX>0);
    assert(resY>0);
    assert(name!="");
    _min = min;
    _max = max;
    _tmin.set(0,0);
    _tmax.set(1,1);
    _resX = resX;
    _resY = resY;
    _isVolume = true;
    buildMesh(mat);
}
//-----------------------------------------------------------------------------
//! SLRectangle ctor with min & max corners and its resolutions 
SLRectangle::SLRectangle(SLVec2f min, SLVec2f max,
                         SLVec2f tmin, SLVec2f tmax,
                         SLuint resX, SLuint resY, 
                         SLstring name,
                         SLMaterial* mat) : SLMesh(name) 
{
    assert(min!=max);
    assert(tmin!=tmax);
    assert(resX>0);
    assert(resY>0);
    assert(name!="");
    _min = min;
    _max = max;
    _tmin = tmin;
    _tmax = tmax;
    _resX = resX;
    _resY = resY;
    _isVolume = true;
    buildMesh(mat);
}
//-----------------------------------------------------------------------------
//! SLRectangle::buildMesh fills in the underlying arrays from the SLMesh object
void SLRectangle::buildMesh(SLMaterial* material)
{  
    deleteData();
   
    // Check max. allowed no. of verts
    SLuint uIntNumV64 = (_resX+1) * (_resY+1);
    if (uIntNumV64 > UINT_MAX)
        SL_EXIT_MSG("SLMesh supports max. 2^32 vertices.");

    // allocate new arrays of SLMesh
    numV = (_resX+1) * (_resY+1);
    numI = _resX * _resY * 2 * 3;
    P = new SLVec3f[numV];
    N = new SLVec3f[numV];
    Tc = new SLVec2f[numV];

    if (uIntNumV64 < 65535)
         I16 = new SLushort[numI];
    else I32 = new SLuint[numI];
   
    // Calculate normal from the first 3 corners
    SLVec3f maxmin(_max.x, _min.y, 0);
    SLVec3f minmax(_min.x, _max.y, 0);
    SLVec3f e1(maxmin - _min);
    SLVec3f e2(minmax - _min);
    SLVec3f curN(e1^e2);
    curN.normalize();
   
    //Set one default material index
    mat = material;
   
    // define delta vectors dX & dY and deltas for texCoord dS,dT
    SLVec3f dX = e1 / (SLfloat)_resX;
    SLVec3f dY = e2 / (SLfloat)_resY;
    SLfloat dS = (_tmax.x - _tmin.x) / (SLfloat)_resX;
    SLfloat dT = (_tmax.y - _tmin.y) / (SLfloat)_resY;

    // Build vertex data
    SLuint i = 0;
    for (SLuint y=0; y<=_resY; ++y)
    {  
        SLVec3f curV = _min;
        SLVec2f curT = _tmin;
        curV   += (SLfloat)y*dY;
        curT.y += (SLfloat)y*dT;

        for (SLuint x=0; x<=_resX; ++x, ++i)
        {  
            P[i]  = curV;
            Tc[i] = curT;
            N[i]  = curN;
            curV   += dX;
            curT.x += dS;
        }      
    }
   
    // Build face vertex indexes
    if (I16)
    {
        SLuint v = 0, i = 0; //index for vertices and indexes
        for (SLuint y=0; y<_resY; ++y)
        {  
            for (SLuint x=0; x<_resX; ++x, ++v)
            {  // triangle 1
                I16[i++] = v;
                I16[i++] = v+_resX+2;
                I16[i++] = v+_resX+1;

                // triangle 2
                I16[i++] = v;
                I16[i++] = v+1;
                I16[i++] = v+_resX+2;
            }      
            v++;
        }
    } else
    {
        SLuint v = 0, i = 0; //index for vertices and indexes
        for (SLuint y=0; y<_resY; ++y)
        {  
            for (SLuint x=0; x<_resX; ++x, ++v)
            {  // triangle 1
                I32[i++] = v;
                I32[i++] = v+_resX+2;
                I32[i++] = v+_resX+1;

                // triangle 2
                I32[i++] = v;
                I32[i++] = v+1;
                I32[i++] = v+_resX+2;
            }      
            v++;
        }
    }
}
//-----------------------------------------------------------------------------
