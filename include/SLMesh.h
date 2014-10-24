//#############################################################################
//  File:      SLMesh.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLMESH_H
#define SLMESH_H

#include <stdafx.h>
#include <SLAABBox.h>
#include <SLGLBuffer.h>

class SLSceneView;
class SLNode;
class SLAccelStruct;
struct SLNodeStats;
class SLMaterial;
class SLRay;

//-----------------------------------------------------------------------------
//!The SLMesh class represents a triangle or line mesh object w. a index
/*!
The SLMesh class represents a single GL_TRIANGLES or GL_LINES mesh object. 
The vertex attributes are stored in arrays with equal number (numV) of elements:
\n P (vertex position)
\n N (vertex normals)
\n C (vertex color)
\n Tc (vertex texture coordinates) optional
\n T (vertex tangents) optional
\n I16 holds the unsigned short vertex indexes.
\n I32 holds the unsigned int vertex indexes.
\n\n
A mesh uses only one material referenced by the SLMesh::_mat pointer.
For each attribute an array vertex buffer object (VBO) is used and encapsulated
in SLGLBuffer.
*/      
class SLMesh : public SLObject
{   
    public:                    
                        SLMesh         (SLstring name = "Mesh");
                       ~SLMesh         ();
               
virtual void            init           (SLNode* node);
virtual void            draw           (SLSceneView* sv, SLNode* node);
        void            addStats       (SLNodeStats &stats);
virtual void            buildAABB      (SLAABBox &aabb, SLMat4f wmNode);
        SLbool          hit            (SLRay* ray, SLNode* node);               
virtual void            preShade       (SLRay* ray);
               
        void            deleteData     ();
virtual void            calcNormals    ();
        void            calcTangents   ();
virtual void            calcMinMax     ();
        void            calcCenterRad  (SLVec3f& center, SLfloat& radius);
        SLbool          hitTriangleOS  (SLRay* ray, SLNode* node, SLuint iT);

        SLPrimitive     primitive      (){return _primitive;}
                               
        SLVec3f*        P;          //!< Array of vertex positions
        SLVec3f*        N;          //!< Array of vertex normals (opt.)
        SLCol4f*        C;          //!< Array of vertex colors (opt.)
        SLVec2f*        Tc;         //!< Array of vertex tex. coords. (opt.)
        SLVec4f*        T;          //!< Array of vertex tangents (opt.)
        SLushort*       I16;        //!< Array of vertex indexes 16 bit
        SLuint*         I32;        //!< Array of vertex indexes 32 bit
        SLuint          numV;       //!< Number of elements in P, N, C, T & Tc   
        SLuint          numI;       //!< Number of elements in I16 or I32
        SLMaterial*     mat;        //!< Pointer to the material
        SLVec3f         minP;       //!< min. vertex in OS
        SLVec3f         maxP;       //!< max. vertex in OS
   
    protected:
        SLGLState*      _stateGL;   //!< Pointer to the global SLGLState instance

        SLPrimitive     _primitive; //!< Primitive type (default triangles)

        SLGLBuffer      _bufP;      //!< Buffer for vertex positions
        SLGLBuffer      _bufN;      //!< Buffer for vertex normals
        SLGLBuffer      _bufC;      //!< Buffer for vertex colors
        SLGLBuffer      _bufTc;     //!< Buffer for vertex texcoords
        SLGLBuffer      _bufT;      //!< Buffer for vertex tangents
        SLGLBuffer      _bufI;      //!< Buffer for vertex indexes
               
        SLGLBuffer      _bufN2;     //!< Buffer for normal line rendering
        SLGLBuffer      _bufT2;     //!< Buffer for tangent line rendering
               
        SLbool          _isVolume;  //!< Flag for RT if mesh is a closed volume
               
        SLAccelStruct*  _accelStruct;   //!< KD-tree or uniform grid
};
//-----------------------------------------------------------------------------
typedef std::vector<SLMesh*>  SLVMesh;
//-----------------------------------------------------------------------------
#endif //SLMESH_H

