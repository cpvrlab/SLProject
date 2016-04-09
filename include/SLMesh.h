//#############################################################################
//  File:      SLMesh.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLMESH_H
#define SLMESH_H

#include <stdafx.h>
#include <SLAABBox.h>
#include <SLGLVertexArray.h>
#include <SLEnums.h>

class SLSceneView;
class SLNode;
class SLAccelStruct;
struct SLNodeStats;
class SLMaterial;
class SLRay;
class SLSkeleton;

/* Problems with the current SLMesh class:
    1.  Too tightly coupled with SLMaterial.
        e.x.:   SLMesh might need a different combination of vertex and fragment programs
                depending on its own data. If it is an animated mesh it needs a vertex program
                that supports GPU skinning. Then we also can choose between per vertex and per
                fragment lighting.

                For the old model it was somewhat ok to specify per vertex/fragment lighting in 
                the material, but specifying a skinning shader in the material doesn't seem right.
*/

//-----------------------------------------------------------------------------
//!An SLMesh object is a triangulated mesh that is drawn with one draw call.
/*!
The SLMesh class represents a single GL_TRIANGLES or GL_LINES mesh object. The
mesh object is drawn with one draw call using the vertex indices in I16 or I32.
The vertex attributes are stored in vectors with equal number of elements:
\n P (vertex position, mandatory)
\n N (vertex normals)
\n C (vertex color)
\n Tc (vertex texture coordinates) optional
\n T (vertex tangents) optional
\n Ji (vertex joint index) optional
\n Jw (vertex joint weights) optional
\n I16 holds the unsigned short vertex indices.
\n I32 holds the unsigned int vertex indices.
\n
\n
The normals of a vertex are automatically calculated in the method calcNormals()
by averaging the face normals of the adjacent triangles. A vertex has always
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
averaged because its position is only once in the vector P. On the other hand
are the vertices of the hard edges in the front of the sphere doubled.
\n
\image HTML sharpAndSmoothEdges.png
\n
\n The following the example creates the box with 24 vertices:
\n The vertex positions and normals in P and N:
\n P.size = 24
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
\n The vertex indices in I16:
\n I16[] = {0,1,2, 0,2,3,
\n          4,5,6, 4,6,7,
\n          8,9,10, 8,10,11,
\n          12,13,14, 12,14,15,
\n          16,17,18, 16,18,19,
\n          20,21,22, 20,22,23}
\n
\image HTML boxVertices.png
\n
All vertex attributes are added to the vertex array object _vao (SLVertexArray).<br>
All arrays remain in the main memory for ray tracing.
A mesh uses normally only one material referenced by the SLMesh::mat pointer.
\n
\n
If a mesh is associated with a skeleton all its vertices and normals are
transformed every frame by the joint weights. Every vertex of a mesh has
weights for four joints by which it can be influenced. This transform is
called skinning and can be done in CPU in the method transformSkin or by
a vertex shader. If the skinning is done on CPU two additional arrays
(_finalP and _finalN) for the transformed vertices and normals are needed.
*/      
class SLMesh : public SLObject
{   
    public:                    
                            SLMesh          (SLstring name = "Mesh");
                           ~SLMesh          ();
               
    virtual void            init            (SLNode* node);
    virtual void            draw            (SLSceneView* sv, SLNode* node);
            void            addStats        (SLNodeStats &stats);
    virtual void            buildAABB       (SLAABBox &aabb, SLMat4f wmNode);
            void            updateAccelStruct();
            SLbool          hit             (SLRay* ray, SLNode* node);               
    virtual void            preShade        (SLRay* ray);
               
            void            deleteData      ();
    virtual void            calcNormals     ();
            void            calcTangents    ();
            void            calcTex3DMatrix (SLNode* node);
    virtual void            calcMinMax      ();
            void            calcCenterRad   (SLVec3f& center, SLfloat& radius);
            SLbool          hitTriangleOS   (SLRay* ray, SLNode* node, SLuint iT);
            void            useHalfFloats   (SLbool useHalf);

            SLGLPrimitiveType primitive       (){return _primitive;}
            void            transformSkin   ();
            void            skinMethod      (SLSkinMethod method);
            SLSkinMethod    skinMethod      () const { return _skinMethod; }
            void            skeleton        (SLSkeleton* skel) { _skeleton = skel; }
      const SLSkeleton*     skeleton        () const { return _skeleton; }
            SLbool          addWeight       (SLint vertId, SLuint jointId, SLfloat weight);
            SLuint          numI            () {return (SLuint)(I16.size() ? I16.size() : I32.size());}
        
            // getter for position and normal data for rendering
            SLVec3f         finalP          (SLuint i) {return _finalP->operator[](i);}
            SLVec3f         finalN          (SLuint i) {return _finalN->operator[](i);}

            // temporary software skinning buffers
            SLVVec3f        skinnedP;       //!< Vector for CPU skinned vertex positions
            SLVVec3f        skinnedN;       //!< Vector for CPU skinned vertex normals 

            SLVVec3f        P;              //!< Vector for vertex positions
            SLVVec3f        N;              //!< Vector for vertex normals (opt.)
            SLVVec2f        Tc;             //!< Vector of vertex tex. coords. (opt.)
            SLVCol4f        C;              //!< Vector of vertex colors (opt.)
            SLVVec4f        T;              //!< Vector of vertex tangents (opt.)
            SLVVec4f        Ji;             //!< Vector of per vertex joint ids (opt.)
            SLVVec4f        Jw;             //!< Vector of per vertex joint weights (opt.)
            SLVushort       I16;            //!< Vector of vertex indices 16 bit
            SLVuint         I32;            //!< Vector of vertex indices 32 bit

            SLMaterial*     mat;            //!< Pointer to the inside material
            SLMaterial*     matOut;         //!< Pointer to the outside material
            SLVec3f         minP;           //!< min. vertex in OS
            SLVec3f         maxP;           //!< max. vertex in OS
   
    protected:
            SLGLState*          _stateGL;       //!< Pointer to the global SLGLState instance
            SLGLPrimitiveType   _primitive;     //!< Primitive type (default triangles)

            SLGLVertexArray     _vao;           //!< OpenGL Vertex Array Object for drawing
            SLGLVertexArrayExt  _vaoN;          //!< OpenGL VAO for optional normal drawing
            SLGLVertexArrayExt  _vaoT;          //!< OpenGL VAO for optional tangent drawing
            SLGLVertexArrayExt  _vaoS;          //!< OpenGL VAO for optional selection drawing
            SLbool              _useHalf;       //!< Use half floats for N,T,C, Tc,Ji & Jw
               
            SLbool              _isVolume;      //!< Flag for RT if mesh is a closed volume
            SLAccelStruct*      _accelStruct;           //!< KD-tree or uniform grid
            SLbool              _accelStructOutOfDate;  //!< flag id accel.struct needs update

            SLSkinMethod        _skinMethod;    //!< CPU or GPU skinning method
            SLSkeleton*         _skeleton;      //!< the skeleton this mesh is bound to
            SLVMat4f            _jointMatrices; //!< joint matrix vector for this mesh
            SLVVec3f*           _finalP;        //!< Pointer to final vertex position vector
            SLVVec3f*           _finalN;        //!< pointer to final vertex normal vector

            void            notifyParentNodesAABBUpdate() const;
};
//-----------------------------------------------------------------------------
typedef std::vector<SLMesh*>  SLVMesh;
//-----------------------------------------------------------------------------
#endif //SLMESH_H

