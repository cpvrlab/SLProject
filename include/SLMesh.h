//#############################################################################
//  File:      SLMesh.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
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
//!An SLMesh object is a triangulated mesh that is drawn with one draw call.
/*!
The SLMesh class represents a single GL_TRIANGLES or GL_LINES mesh object. The
mesh object is drawn with one draw call using the vertex indexes in I16 or I32.
The vertex attributes are stored in arrays with equal number (numV) of elements:
\n P (vertex position)
\n N (vertex normals)
\n C (vertex color)
\n Tc (vertex texture coordinates) optional
\n T (vertex tangents) optional
\n I16 holds the unsigned short vertex indexes.
\n I32 holds the unsigned int vertex indexes.
\n
\n
The normals of a vertex are automatically calculated in the method calcNormals()
by averageing the face normals of the adjacent triangles. A vertex has allways
only <b>one</b> normal and is used for the lighting calculation in the shader
programs. With such averaged normals you can created a interpolated shading on
smooth surfaces such as a sphere.
\n
For objects with sharp edges such as a box you need 4 vertices per box face.
All normals of a face point to the same direction. This means, that you have
three times the same vertex position but with different normals for one corner
of the box.
\n
The following image shows a box with sharp edges and a sphere with mostly
smooth but also 4 sharp edges. The smooth red normal as the top vertex got
averaged because its position is only once in the array P. On the other hand
are the vertices of the hard edges in the front of the sphere doubled.
\n
\image html sharpAndSmoothEdges.png
\n
\n The following the example creates the box with 24 vertices:
\n The vertex positios and normals in P and N:
\n numV = 24
\n P[0] = [1,1,1]   N[0] = [1,0,0]
\n P[1] = [1,0,1]   N[1] = [1,0,0]
\n P[2] = [1,0,0]   N[2] = [1,0,0]
\n P[3] = [1,1,0]   N[3] = [1,0,0]
\n
\n P[4] = [1,1,0]   N[4] = [0,0,-1]
\n P[5] = [1,0,0]   N[5] = [0,0,-1]
\n P[6] = [0,0,0]   N[6] = [0,0,-1]
\n P[7] = [0,1,0]   N[7] = [0,0,-1]
\n
\n P[8] = [0,0,1]   N[8] = [-1,0,0]
\n P[9] = [0,1,1]   N[9] = [-1,0,0]
\n P[10]= [0,1,0]   N[10]= [-1,0,0]
\n P[11]= [0,0,0]   N[11]= [-1,0,0]
\n
\n P[12]= [1,1,1]   N[12]= [0,0,1]
\n P[13]= [0,1,1]   N[13]= [0,0,1]
\n P[14]= [0,0,1]   N[14]= [0,0,1]
\n P[15]= [1,0,1]   N[15]= [0,0,1]
\n
\n P[16]= [1,1,1]   N[16]= [0,1,0]
\n P[17]= [1,1,0]   N[17]= [0,1,0]
\n P[18]= [0,1,0]   N[18]= [0,1,0]
\n P[19]= [0,1,1]   N[19]= [0,1,0]
\n
\n P[20]= [0,0,0]   N[20]= [0,-1,0]
\n P[21]= [1,0,0]   N[21]= [0,-1,0]
\n P[22]= [1,0,1]   N[22]= [0,-1,0]
\n P[23]= [0,0,1]   N[23]= [0,-1,0]
\n
\n The vertex indexes in I16:
\n I16[] = {0,1,2, 0,2,3,
\n          4,5,6, 4,6,7,
\n          8,9,10, 8,10,11,
\n          12,13,14, 12,14,15,
\n          16,17,18, 16,18,19,
\n          20,21,22, 20,22,23}
\n
\image html boxVertices.png
\n
For all arrays a corresponding vertex buffer object (VBO) is created on the
graphic card. All arrays remain in the main memory for ray tracing.
A mesh uses only one material referenced by the SLMesh::_mat pointer.
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

