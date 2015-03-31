//#############################################################################
//  File:      SLRevolver.cpp
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

#include <SLRevolver.h>

//-----------------------------------------------------------------------------
/*!
SLRevolver::SLRevolver ctor for generic revolution object.
*/
SLRevolver::SLRevolver(SLVVec3f revolvePoints, 
                       SLVec3f  revolveAxis,
                       SLint    slices,
                       SLbool   smoothFirst, SLbool   smoothLast, 
                       SLstring name,
                       SLMaterial* mat) : SLMesh(name)
{  
    assert(revolvePoints.size() >= 2 && "Error: Not enough revolve points.");
    assert(revolveAxis!=SLVec3f::ZERO && "Error axis is a zero vector.");
    assert(slices >= 3 && "Error: Not enough slices.");
   
    _revPoints   = revolvePoints;
    _revAxis     = revolveAxis;
    _slices      = slices;
    _smoothFirst = smoothFirst;
    _smoothLast  = smoothLast;
   
    _revAxis.normalize();
   
    buildMesh(mat);
}
//-----------------------------------------------------------------------------
/*!
SLRevolver::buildMesh builds the underlying mesh data structure
*/
void SLRevolver::buildMesh(SLMaterial* material)
{  
    deleteData();
      
    ///////////////////////////////
    // Vertices & Texture coords //
    ///////////////////////////////

    // calculate no. of vertices & allocate arrays for P, Tc & N.
    // On one stack it has one vertex more at the end that is identical with the
    // first vertex of the stack. This is for cylindrical texture mapping where 
    // we need 2 different texture s-coords (0 & 1) at the same point.
    numV = (SLuint)((_slices+1) * _revPoints.size());
    P    = new SLVec3f[numV];
    N    = new SLVec3f[numV];
    Tc   = new SLVec2f[numV];
   
    // calculate length of segments for texture coords
    SLfloat totalLenght = 0;
    std::vector<SLfloat> segments;
    segments.push_back(0);
    for (SLuint r=0; r<_revPoints.size()-1; ++r)
    {   SLVec3f s = _revPoints[r+1] - _revPoints[r];
        SLfloat len = s.length();
        totalLenght += len;
        segments.push_back(len);
    }

    // Normalize segment lenghts for texture coords
    for (auto& segment : segments)
        segment /= totalLenght;
   
    // Texture coordinate
    SLVec2f texCoord(0, 0);          // y is increased by segment[r]
    SLfloat deltaS = 1.0f/_slices;   // increase value for s-tecCoord
   
    // define matrix & angles for rotation
    SLMat4f m;
    SLfloat dPhi = 360.0f/_slices;
     
    // calculate vertices & texture coords for all revolution points
    SLuint iV = 0;
    for (SLuint r=0; r<_revPoints.size(); ++r)
    {   m.identity();
        texCoord.x = 0;
        texCoord.y += segments[r];
        for (SLint s=0; s<=_slices; ++s)
        {  
            if (s==0 || s==_slices)
                 P[iV] = _revPoints[r];
            else P[iV] = m.multVec(_revPoints[r]);

            Tc[iV++] = texCoord;
            m.rotate(dPhi, _revAxis);
            texCoord.x += deltaS;
        }
    }
   
    /////////////////
    //    Faces    //
    /////////////////

    std::vector<SLVec3ui> faces;
   
    // set faces (triangles) for all revolution segments
    SLVec3ui f;
    SLuint iV1, iV2;
    for (SLuint r=0; r<_revPoints.size()-1; ++r)
    {   iV1 =  r    * (_slices+1);
        iV2 = (r+1) * (_slices+1);
      
        // only define faces if neighbouring points are different
        if (_revPoints[r] != _revPoints[r+1])
        {  
            for (SLint s=0; s<_slices; ++s)
            {  
                // Add two triangles if real quad is visible
                // Add upper triangle only iB (or iC) are not on rev. axis
                if (_revAxis.distSquared(P[iV2+s]) > FLT_EPSILON)
                {   f.x = iV1+s;
                    f.y = iV2+s+1;
                    f.z = iV2+s;
                    faces.push_back(f);
                }

                // Add lower triangle only iA (or iB) are not on rev. axis
                if (_revAxis.distSquared(P[iV1+s]) > FLT_EPSILON)
                {   f.x = iV1+s;
                    f.y = iV1+s+1;
                    f.z = iV2+s+1;
                    faces.push_back(f);
                }
            }
        }
    }

    // calculate no. of faces (triangles) & allocate arrays
    numI = (SLuint)faces.size() * 3;
    SLuint i = 0;
    if (numV < 65536)
    {   I16 = new SLushort[numI];
        for(auto face : faces) 
        {   I16[i++] = face.x;
            I16[i++] = face.y;
            I16[i++] = face.z;
        }
    } else
    {   I32 = new SLuint[numI];
        for(auto face : faces) 
        {   I32[i++] = face.x;
            I32[i++] = face.y;
            I32[i++] = face.z;
        }
    }

    //Set one default material index
    mat = material;

    /////////////////
    //   Normals   //
    /////////////////
   
    // Calculate normals with the SLMesh method
    calcNormals();
   
    // correct normals at the first point
    if (_smoothFirst)
    {   for (SLint s=0; s<_slices; ++s)
        {   N[s  ] = -_revAxis;
            N[s+1] = -_revAxis;
        }
    }   
   
    // correct normals at the first point
    if (_smoothLast)
    {   for (SLint s=0; s<_slices; ++s)
        {   N[numV-s-1] = _revAxis;
            N[numV-s-2] = _revAxis;
        }
    }
   
    // correct (smooth) the start normal and the end normal of a stack
    for (SLint r=0; r<_revPoints.size(); ++r)
    {   iV1 =  r * (_slices+1);
        iV2 = iV1 + _slices;
        N[iV1] += N[iV2];
        N[iV1].normalize();
        N[iV2] = N[iV1];
    }  
}
//-----------------------------------------------------------------------------
