//#############################################################################
//  File:      SLTriangle.cpp
//  Author:    Philipp Jüni
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include "SLTriangle.h"

//-----------------------------------------------------------------------------
SLTriangle::SLTriangle(SLMaterial *material, SLstring name, 
                       SLVec3f p0, SLVec3f p1, SLVec3f p2, 
                       SLVec2f t0, SLVec2f t1, SLVec2f t2 ) : SLMesh(name)
{
    p[0] = p0;
    p[1] = p1;
    p[2] = p2;

    t[0] = t0;
    t[1] = t1;
    t[2] = t2;

    mat = material;

    _isVolume = false;

    buildMesh(mat);
}
//-----------------------------------------------------------------------------
//! Builds the mesh by copying the vertex info into the arrays of SLMescj
void SLTriangle::buildMesh(SLMaterial* material)
{
    deleteData();

    numV = 3;                   //!< Number of elements in P, N, C, T & Tc
    numI = 3;                   //!< Number of elements in I16 or I32
    P = new SLVec3f[numV];      //!< Array of vertex positions
    N = new SLVec3f[numV];      //!< Array of vertex normals (opt.)
    Tc = new SLVec2f[numV];     //!< Array of vertex tex. coords. (opt.)
    I16 = new SLushort[numI];   //!< Array of vertex indexes 16 bit

    // vertex positions
    P[0] = p[0];
    P[1] = p[1];
    P[2] = p[2];
    
    // vertex texture coordinates
    Tc[0] = t[0];
    Tc[1] = t[1];
    Tc[2] = t[2];

    // indexes
    I16[0] = 0;
    I16[1] = 1;
    I16[2] = 2;

    // normals    
    SLVec3f n = (p[1] - p[0]) ^ (p[2] - p[0]);
    n.normalize();
    N[0] = n;
    N[1] = n;
    N[2] = n;
}
//-----------------------------------------------------------------------------
