//#############################################################################
//  File:      SLGLVertexArray.h
//  Purpose:   Wrapper class around OpenGL Vertex Array Objects (VAO)
//  Date:      January 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLVERTEXARRAY_H
#define SLGLVERTEXARRAY_H

#include <SLGLEnums.h>
#include <SLGLVertexBuffer.h>

//-----------------------------------------------------------------------------
//! SLGLVertexArray encapsulates the core OpenGL drawing
/*! An SLGLVertexArray instance handles all OpenGL drawing with an OpenGL
 Vertex Array Object (VAO), a vertex buffer objects (VBO) for the attributes
 and an index buffer for element drawing. Attributes can be stored in a float
 VBO of type SLGLVertexBuffer.\n
 VAOs where introduces OpenGL 3.0 and reduce the overhead per draw call.
 All vertex attributes (e.g. position, normals, texture coords, etc.) must be
 float at the input. All float attributes will be in one VBO (_VBOf).
 Vertices can be drawn either directly as in the array (SLGLVertexArray::drawArrayAs)
 or by element (SLGLVertexArray::drawElementsAs) with a separate indices buffer.\n
 The setup of a VAO has multiple steps:\n
 - Define one ore more attributes with SLGLVertexArray::setAttrib.
 - Define the index array for element drawing with SLGLVertexArray::setIndices.
 - Generate the OpenGL VAO and VBO with SLGLVertexArray::generate.\n
 It is important that the data structures passed in SLGLVertexArray::setAttrib and
 SLGLVertexArray::setIndices are still present when generate is called.
 The VAO has no or one active index buffer. For drawArrayAs no indices are needed.
 For drawElementsAs the index buffer is used. For triangle meshes also hard edges
 are generated. Their indices are stored behind the indices of the triangles.
 See SLMesh::computeHardEdgesIndices for more infos on hard edges.
*/
class SLGLVertexArray
{
public:
    SLGLVertexArray();
    ~SLGLVertexArray() { deleteGL(); }

    //! Deletes all vertex array & vertex buffer objects
    void deleteGL();

    //! Clears the attribute definition
    void clearAttribs()
    {
        deleteGL();
        _VBOf.clear();
    }

    //! Returns either the VAO id or the VBO id
    SLint vaoID() const { return _vaoID; }

    //! Returns the TFO id
    SLint tfoID() const { return _tfoID; }

    //! Adds a vertex attribute with data pointer and an element size
    void setAttrib(SLGLAttributeType type,
                   SLint             elementSize,
                   SLint             location,
                   void*             dataPointer,
                   SLGLBufferType    dataType = BT_float);

    //! Adds a vertex attribute with vector of SLuint
    void setAttrib(SLGLAttributeType type,
                   SLint             location,
                   SLVuint*          data) { setAttrib(type, 1, location, &data->operator[](0), BT_uint); }

    //! Adds a vertex attribute with vector of SLfloat
    void setAttrib(SLGLAttributeType type,
                   SLint             location,
                   SLVfloat*         data) { setAttrib(type, 1, location, &data->operator[](0)); }

    //! Adds a vertex attribute with vector of SLVec2f
    void setAttrib(SLGLAttributeType type,
                   SLint             location,
                   SLVVec2f*         data) { setAttrib(type, 2, location, &data->operator[](0)); }

    //! Adds a vertex attribute with vector of SLVec3f
    void setAttrib(SLGLAttributeType type,
                   SLint             location,
                   SLVVec3f*         data) { setAttrib(type, 3, location, &data->operator[](0)); }

    //! Adds a vertex attribute with vector of SLVec4f
    void setAttrib(SLGLAttributeType type,
                   SLint             location,
                   SLVVec4f*         data) { setAttrib(type, 4, location, &data->operator[](0)); }

    //! Adds the index array for indexed element drawing
    void setIndices(SLuint         numIndicesElements,
                    SLGLBufferType indexDataType,
                    void*          indexDataElements,
                    SLuint         numIndicesEdges = 0,
                    void*          indexDataEdges  = nullptr);

    //! Adds the index array for indexed element drawing with a vector of ubyte
    void setIndices(SLVubyte* indicesElements,
                    SLVubyte* indicesEdges = nullptr)
    {
        setIndices((SLuint)indicesElements->size(),
                   BT_ubyte,
                   (void*)&indicesElements->operator[](0),
                   indicesEdges ? (SLuint)indicesEdges->size() : 0,
                   indicesEdges && indicesEdges->size() ? (void*)&indicesEdges->operator[](0) : nullptr);
    };

    //! Adds the index array for indexed element drawing with a vector of ushort
    void setIndices(SLVushort* indicesElements,
                    SLVushort* indicesEdges = nullptr)
    {
        setIndices((SLuint)indicesElements->size(),
                   BT_ushort,
                   (void*)&indicesElements->operator[](0),
                   indicesEdges ? (SLuint)indicesEdges->size() : 0,
                   indicesEdges && indicesEdges->size() ? (void*)&indicesEdges->operator[](0) : nullptr);
    };

    //! Adds the index array for indexed element drawing with a vector of uint
    void setIndices(SLVuint* indicesElements,
                    SLVuint* indicesEdges = nullptr)
    {
        setIndices((SLuint)indicesElements->size(),
                   BT_uint,
                   (void*)&indicesElements->operator[](0),
                   indicesEdges ? (SLuint)indicesEdges->size() : 0,
                   indicesEdges && indicesEdges->size() ? (void*)&indicesEdges->operator[](0) : nullptr);
    }

    //! Updates a specific vertex attribute in the VBO
    void updateAttrib(SLGLAttributeType type,
                      SLint             elementSize,
                      void*             dataPointer);

    //! Updates a specific vertex attribute in the VBO
    void updateAttrib(SLGLAttributeType type,
                      SLVuint*          data) { updateAttrib(type, 1, (void*)&data->operator[](0)); }

    //! Updates a specific vertex attribute in the VBO
    void updateAttrib(SLGLAttributeType type,
                      SLVfloat*         data) { updateAttrib(type, 1, (void*)&data->operator[](0)); }

    //! Updates a specific vertex attribute in the VBO
    void updateAttrib(SLGLAttributeType type,
                      SLVVec2f*         data) { updateAttrib(type, 2, (void*)&data->operator[](0)); }

    //! Updates a specific vertex attribute in the VBO
    void updateAttrib(SLGLAttributeType type,
                      SLVVec3f*         data) { updateAttrib(type, 3, (void*)&data->operator[](0)); }

    //! Updates a specific vertex attribute in the VBO
    void updateAttrib(SLGLAttributeType type,
                      SLVVec4f*         data) { updateAttrib(type, 4, (void*)&data->operator[](0)); }

    //! Generates the VA & VB objects for a NO. of vertices
    void generate(SLuint          numVertices,
                  SLGLBufferUsage usage             = BU_static,
                  SLbool          outputInterleaved = true);

    //! Generates the VA & VB & TF objects
    void generateTF(SLuint          numVertices,
                    SLGLBufferUsage usage             = BU_static,
                    SLbool          outputInterleaved = true);

    //! Begin transform feedback
    void beginTF(SLuint tfoID);

    //! End transform feedback
    void endTF();

    //! Draws the VAO by element indices with a primitive type
    void drawElementsAs(SLGLPrimitiveType primitiveType,
                        SLuint            numIndexes       = 0,
                        SLuint            indexOffsetBytes = 0);

    //! Draws the VAO as an array with a primitive type
    void drawArrayAs(SLGLPrimitiveType primitiveType,
                     SLint             firstVertex   = 0,
                     SLsizei           countVertices = 0);

    //! Draws the hard edges of the VAO with the edge indices
    void drawEdges(SLCol4f color, SLfloat lineWidth = 1.0f);

    // Some getters
    SLuint numVertices() const { return _numVertices; }
    SLuint numIndicesElements() const { return _numIndicesElements; }
    SLuint numIndicesEdges() const { return _numIndicesEdges; }

    // Some statistics
    static SLuint totalDrawCalls;          //! static total no. of draw calls
    static SLuint totalPrimitivesRendered; //! static total no. of primitives rendered

protected:
    SLuint           _vaoID;              //! OpenGL id of vertex array object
    SLuint           _tfoID;              //! OpenGL id of transform feedback object
    SLuint           _numVertices;        //! NO. of vertices in array
    SLGLVertexBuffer _VBOf;               //! Vertex buffer object for float attributes
    SLuint           _idVBOIndices;       //! OpenGL id of index vbo
    SLuint           _numIndicesElements; //! NO. of vertex indices in array for triangles, lines or points
    void*            _indexDataElements;  //! Pointer to index data for elements
    SLuint           _numIndicesEdges;    //! NO. of vertex indices in array for hard edges
    void*            _indexDataEdges;     //! Pointer to index data for hard edges
    SLGLBufferType   _indexDataType;      //! index data type (ubyte, ushort, uint)
};
//-----------------------------------------------------------------------------

#endif
