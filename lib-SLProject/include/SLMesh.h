//#############################################################################
//  File:      SLMesh.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLMESH_H
#define SLMESH_H

#include <SLAABBox.h>
#include <SLEnums.h>
#include <SLGLVertexArray.h>
#include <SLObject.h>
#include <SLOptixAccelStruct.h>
#include <SLOptixDefinitions.h>

class SLSceneView;
class SLNode;
class SLAccelStruct;
struct SLNodeStats;
class SLMaterial;
class SLRay;
class SLSkeleton;
class SLGLState;
class SLGLProgram;
class SLAssetManager;

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
\n Ji (vertex joint index) optional 2D vector
\n Jw (vertex joint weights) optional 2D vector
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
weights for 1-n joints by which it can be influenced. This transform is
called skinning and is done in CPU in the method transformSkin. The final
transformed vertices and normals are stored in _finalP and _finalN.
*/

class SLMesh : public SLObject
#ifdef SL_HAS_OPTIX
  , public SLOptixAccelStruct
#endif

{
public:
    explicit SLMesh(SLAssetManager* assetMgr,
                    const SLstring& name = "Mesh");
    ~SLMesh() override;

    virtual void init(SLNode* node);
    virtual void draw(SLSceneView* sv, SLNode* node);
    void         drawIntoDepthBuffer(SLSceneView* sv, SLNode* node, SLMaterial* depthMat);
    void         addStats(SLNodeStats& stats);
    virtual void buildAABB(SLAABBox& aabb, const SLMat4f& wmNode);
    void         updateAccelStruct();
    SLbool       hit(SLRay* ray, SLNode* node);
    virtual void preShade(SLRay* ray);

    void         deleteData();
    void         deleteSelected(SLNode* node);
    void         deleteUnused();
    void         calcTex3DMatrix(SLNode* node);
    virtual void calcMinMax();
    virtual void calcNormals();
    void         calcCenterRad(SLVec3f& center, SLfloat& radius);
    SLbool       hitTriangleOS(SLRay* ray, SLNode* node, SLuint iT);
    void         generateVAO(SLGLVertexArray& vao);
    void         transformSkin(const std::function<void(SLMesh*)>& cbInformNodes);
    void         deselectPartialSelection();

#ifdef SL_HAS_OPTIX
    void                allocAndUploadData();
    void                uploadData();
    virtual void        createMeshAccelerationStructure();
    virtual void        updateMeshAccelerationStructure();
    virtual ortHitData  createHitData();
    unsigned int        sbtIndex() const { return _sbtIndex; }
    static unsigned int meshIndex;
#endif

    // Getters
    SLMaterial*       mat() const { return _mat; }
    SLMaterial*       matOut() const { return _matOut; }
    SLGLPrimitiveType primitive() const { return _primitive; }
    const SLSkeleton* skeleton() const { return _skeleton; }
    SLuint            numI() const { return (SLuint)(!I16.empty() ? I16.size() : I32.size()); }
    SLGLVertexArray&  vao() { return _vao; }
    SLbool            isSelected() const { return _isSelected; }

    // Setters
    void mat(SLMaterial* m) { _mat = m; }
    void matOut(SLMaterial* m) { _matOut = m; }
    void primitive(SLGLPrimitiveType pt) { _primitive = pt; }
    void skeleton(SLSkeleton* skel) { _skeleton = skel; }
    void isSelected(bool isSelected) { _isSelected = isSelected; }

    // getter for position and normal data for rendering
    SLVec3f finalP(SLuint i) { return _finalP->operator[](i); }
    SLVec3f finalN(SLuint i) { return _finalN->operator[](i); }

    // temporary software skinning buffers
    SLVVec3f skinnedP; //!< Vector for CPU skinned vertex positions
    SLVVec3f skinnedN; //!< Vector for CPU skinned vertex normals

    SLVVec3f  P;  //!< Vector for vertex positions                   layout (location = 0)
    SLVVec3f  N;  //!< Vector for vertex normals (opt.)              layout (location = 1)
    SLVVec2f  Tc; //!< Vector of vertex tex. coords. (opt.)          layout (location = 2)
    SLVCol4f  C;  //!< Vector of vertex colors (opt.)                layout (location = 3)
    SLVVec4f  T;  //!< Vector of vertex tangents (opt.)              layout (location = 4)
    SLVVuchar Ji; //!< 2D Vector of per vertex joint ids (opt.)      layout (location = 5) (never used)
    SLVVfloat Jw; //!< 2D Vector of per vertex joint weights (opt.)  layout (location = 6) (never used)

    SLVushort I16;  //!< Vector of vertex indices 16 bit
    SLVuint   I32;  //!< Vector of vertex indices 32 bit
    SLVuint   IS32; //!< Vector of rectangle selected vertex indices 32 bit

    SLVec3f minP; //!< min. vertex in OS
    SLVec3f maxP; //!< max. vertex in OS

private:
    void calcTangents();
    void handleRectangleSelection(SLSceneView* sv,
                                  SLGLState*   stateGL,
                                  SLNode*      node);
    void drawSelectedVertices();

protected:
    SLGLPrimitiveType  _primitive;  //!< Primitive type (default triangles)
    SLMaterial*        _mat;        //!< Pointer to the inside material
    SLMaterial*        _matOut;     //!< Pointer to the outside material
    SLGLVertexArray    _vao;        //!< OpenGL Vertex Array Object for drawing
    SLGLVertexArrayExt _vaoN;       //!< OpenGL VAO for optional normal drawing
    SLGLVertexArrayExt _vaoT;       //!< OpenGL VAO for optional tangent drawing
    SLGLVertexArrayExt _vaoS;       //!< OpenGL VAO for optional selection drawing
    SLbool             _isSelected; //!< flag if mesh is partially of fully selected

#ifdef SL_HAS_OPTIX
    SLCudaBuffer<SLVec3f>  _vertexBuffer;
    SLCudaBuffer<SLVec3f>  _normalBuffer;
    SLCudaBuffer<SLVec2f>  _textureBuffer;
    SLCudaBuffer<SLushort> _indexShortBuffer;
    SLCudaBuffer<SLuint>   _indexIntBuffer;
    unsigned int           _sbtIndex;
#endif

    SLbool         _isVolume;             //!< Flag for RT if mesh is a closed volume
    SLAccelStruct* _accelStruct;          //!< KD-tree or uniform grid
    SLbool         _accelStructOutOfDate; //!< flag id accel.struct needs update

    SLSkeleton* _skeleton;      //!< the skeleton this mesh is bound to
    SLVMat4f    _jointMatrices; //!< joint matrix vector for this mesh
    SLVVec3f*   _finalP;        //!< Pointer to final vertex position vector
    SLVVec3f*   _finalN;        //!< pointer to final vertex normal vector

    //void notifyParentNodesAABBUpdate(std::function<void(void)> cbInformNodes) const;
};
//-----------------------------------------------------------------------------
typedef vector<SLMesh*> SLVMesh;
//-----------------------------------------------------------------------------
#endif //SLMESH_H
