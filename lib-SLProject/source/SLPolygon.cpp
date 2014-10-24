//#############################################################################
//  File:      SLPolygon.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLPolygon.h>

//-----------------------------------------------------------------------------
//! SLPolygon ctor with corner points array 
SLPolygon::SLPolygon(SLVVec3f corner, SLstring name, SLMaterial* mat) 
          :SLMesh(name) 
{
    assert(corner.size()>2);
    _corner = corner;
    buildMesh(mat);
}
//-----------------------------------------------------------------------------
//! SLPolygon ctor with corner points and its texture coords array 
SLPolygon::SLPolygon(SLVVec3f corner, 
                     SLVVec2f texCoord,
                     SLstring name, 
                     SLMaterial*   mat) :SLMesh(name)   
{
    assert(corner.size()>2 && texCoord.size()==corner.size());
    _corner = corner;
    _texCoord = texCoord;
    buildMesh(mat);
}
//-----------------------------------------------------------------------------
//! SLPolygon ctor for centered light rectangle in x/y-plane w. N=-z 
SLPolygon::SLPolygon(SLfloat  width, 
                     SLfloat  height,
                     SLstring name, 
                     SLMaterial*   mat) :SLMesh(name)  
{
    assert(width>0 && height>0);
    SLfloat hw = width  * 0.5f;
    SLfloat hh = height * 0.5f;
    _corner.push_back(SLVec3f( hw, hh));
    _corner.push_back(SLVec3f( hw,-hh));
    _corner.push_back(SLVec3f(-hw,-hh));
    _corner.push_back(SLVec3f(-hw, hh));
    buildMesh(mat);
}
//-----------------------------------------------------------------------------
//! SLPolygon::buildMesh fills in the underlying arrays from the SLMesh object
void SLPolygon::buildMesh(SLMaterial* material)
{  
    _isVolume = false;
   
    deleteData();
   
    // Check max. allowed no. of verts
    if (_corner.size() >= 65535) 
        SL_EXIT_MSG("SLPolygon::buildMesh: NO. of vertices exceeds the maximum (65535) allowed.");

    // allocate new arrays of SLMesh
    numV = (SLuint)_corner.size();
    numI = (numV - 2) * 3 ;
    P = new SLVec3f[numV];
    N = new SLVec3f[numV];
    if (_texCoord.size()) Tc = new SLVec2f[numV];
    I16 = new SLushort[numI];
   
    // Calculate normal from the first 3 corners
    SLVec3f v1(_corner[0]-_corner[1]);
    SLVec3f v2(_corner[0]-_corner[2]);
    SLVec3f n(v1^v2);
    n.normalize();
   
    //Set one default material index
    mat = material;
   
    //Copy vertices and normals
    for (SLushort i=0; i<numV; ++i)
    {   P[i] = _corner[i];
        N[i] = n;
        if (Tc) Tc[i] = _texCoord[i];
    }
   
    // Build face vertex indexes
    for (SLuint f=0; f<_corner.size()-2; ++f) 
    {   SLuint i = f * 3;
        I16[i  ] = 0;
        I16[i+1] = f+1;
        I16[i+2] = f+2;
    }
}
//-----------------------------------------------------------------------------
