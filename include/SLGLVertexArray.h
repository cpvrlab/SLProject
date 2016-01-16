//#############################################################################
//  File:      SLGLVertexArray.h
//  Purpose:   Wrapper class around OpenGL Vertex Array Objects (VAO) 
//  Author:    Marcus Hudritsch
//  Date:      January 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLVERTEXARRAY_H
#define SLGLVERTEXARRAY_H

#include <stdafx.h>

//-----------------------------------------------------------------------------
//! Enumeration for buffer data types
enum SLBufferType
{   BT_float  = GL_FLOAT,          //!< vertex attributes (position, normals)
    BT_ubyte  = GL_UNSIGNED_BYTE,  //!< vertex index type (0-2^8)
    BT_ushort = GL_UNSIGNED_SHORT, //!< vertex index type (0-2^16)
    BT_uint   = GL_UNSIGNED_INT    //!< vertex index type (0-2^32)
};
//-----------------------------------------------------------------------------
//! Enumeration for buffer usage types also supported by OpenGL ES
enum SLBufferUsage
{   BU_static  = GL_STATIC_DRAW,    //!< Buffer will be modified once and used many times.
    BU_stream  = GL_STREAM_DRAW,    //!< Buffer will be modified once and used at most a few times.
    BU_dynamic = GL_DYNAMIC_DRAW,   //!< Buffer will be modified repeatedly and used many times.
};
//-----------------------------------------------------------------------------
// Enumeration for OpenGL primitive types
enum SLPrimitiveType
{   PT_points        = GL_POINTS,
    PT_lines         = GL_LINES,
    PT_lineLoop      = GL_LINE_LOOP,
    PT_lineStrip     = GL_LINE_STRIP,
    PT_triangles     = GL_TRIANGLES,
    PT_triangleStrip = GL_TRIANGLE_STRIP,
    PT_triangleFan   = GL_TRIANGLE_FAN
};
//-----------------------------------------------------------------------------
//! Enumeration for float vertex attribute types
enum SLVertexAttribType
{   VAT_position,    //!< Vertex position as a 2, 3 or 4 component vectors
    VAT_normal,      //!< Vertex normal as a 3 component vector
    VAT_texCoord,    //!< Vertex texture coordinate as 2 component vector
    VAT_tangent,     //!< Vertex tangent as a 4 component vector (see SLMesh) 
    VAT_jointWeight, //!< Vertex joint weight for vertex skinning
    VAT_jointIndex,  //!< Vertex joint id for vertex skinning
    VAT_color,       //!< Vertex color as 3 or 4 component vector
    VAT_custom0,     //!< Custom vertex attribute 0
    VAT_custom1,     //!< Custom vertex attribute 1
    VAT_custom2,     //!< Custom vertex attribute 2
    VAT_custom3,     //!< Custom vertex attribute 3
    VAT_custom4,     //!< Custom vertex attribute 4
    VAT_custom5,     //!< Custom vertex attribute 5
    VAT_custom6,     //!< Custom vertex attribute 6
    VAT_custom7,     //!< Custom vertex attribute 7
    VAT_custom8,     //!< Custom vertex attribute 8
    VAT_custom9      //!< Custom vertex attribute 0
};
//-----------------------------------------------------------------------------
//! Struct for vertex attribute information
struct SLVertexAttrib
{   SLVertexAttribType  type;           //!< type of vertex attribute
    SLint               elementSize;    //!< size of attribute element (SLVec3f has 3)
    SLuint              offsetBytes;    //!< offset of the attribute data in the buffer
    SLuint              bufferSizeBytes;//!< size of the attribute part in the buffer
    void*               dataPointer;    //!< pointer to the attributes source data
    SLint               location;       //!< GLSL input variable location index
};
//-----------------------------------------------------------------------------
typedef vector<SLVertexAttrib>  SLVVertexAttrib;
//-----------------------------------------------------------------------------




//-----------------------------------------------------------------------------
//! SLGLVertexArray encapsulates the core OpenGL drawing
/*! An SLGLVertexArray instance handles all OpenGL drawing with an OpenGL 
Vertex Array Object (VAO ) and a float Vertex Buffer Object (VBO) for all
attributes and an index buffer for element drawing.\n 
VAOs where introduces OpenGL 3.0 and reduce the overhead per draw call. 
All vertex attributes (e.g. position, normals, texture coords, etc.) are float
and are stored in one big VBO. They can be in sequential order (first all 
positions, then all normals, etc.) or interleaved (all attributes together for
one vertex).\n
Vertices can be drawn either directly as in the array (SLGLVertexArray::drawArrayAs) 
or by element (SLGLVertexArray::drawElementsAs) with a separate indices buffer.\n
The setup of a VAO has multiple steps:\n
- Define one ore more attributes with SLGLVertexArray::setAttrib.
- Define the index array for element drawing with SLGLVertexArray::setIndices.
- Generate the OpenGL VAO and VBO with SLGLVertexArray::generate.\n
It is important that the data structures passed in SLGLVertexArray::setAttrib and 
SLGLVertexArray::setIndices are still present when generate is called.
*/
class SLGLVertexArray
{
    public:
                    SLGLVertexArray     ();
                   ~SLGLVertexArray     () {deleteGL();}
        
        //! Deletes all vertex array & vertex buffer objects
        void        deleteGL            ();

        //! Clears the attribute definition
        void        clearAttribs        () {deleteGL(); _attribs.clear();}

        //! Returns either the VAO id or the VBO id
        SLint       id                  () {return _glHasVAO?_idVAO:_idVBOAttribs;}

        //! Returns the vector index if a vertex attribute exists otherwise -1
        SLint       attribIndex         (SLVertexAttribType type);

        //! Adds a vertex attribute with data pointer and an element size
        void        setAttrib           (SLVertexAttribType type, 
                                         SLint elementSize, 
                                         SLint location, 
                                         void* dataPointer);

        //! Adds a vertex attribute with vector of SLfloat
        void        setAttrib           (SLVertexAttribType type,
                                         SLint location, 
                                         SLVfloat& data) {setAttrib(type, 1, location, (void*)&data[0]);}

        //! Adds a vertex attribute with vector of SLVec2f
        void        setAttrib           (SLVertexAttribType type,
                                         SLint location, 
                                         SLVVec2f& data) {setAttrib(type, 2, location, (void*)&data[0]);}

        //! Adds a vertex attribute with vector of SLVec3f
        void        setAttrib           (SLVertexAttribType type,
                                         SLint location, 
                                         SLVVec3f& data) {setAttrib(type, 3, location, (void*)&data[0]);}

        //! Adds a vertex attribute with vector of SLVec4f
        void        setAttrib           (SLVertexAttribType type,
                                         SLint location, 
                                         SLVVec4f& data) {setAttrib(type, 4, location, (void*)&data[0]);}
        
        //! Adds the index array for indexed element drawing
        void        setIndices          (SLuint numIndices,
                                         SLBufferType indexDataType,
                                         void* dataPointer);
        
        //! Adds the index array for indexed element drawing with a vector of ubyte
        void        setIndices          (SLVubyte& indices) {setIndices((SLuint)indices.size(), 
                                                                        BT_ubyte, 
                                                                        (void*)&indices[0]);}
        
        //! Adds the index array for indexed element drawing with a vector of ushort
        void        setIndices          (SLVushort& indices) {setIndices((SLuint)indices.size(), 
                                                                         BT_ushort, 
                                                                         (void*)&indices[0]);}
        
        //! Adds the index array for indexed element drawing with a vector of uint
        void        setIndices          (SLVuint& indices) {setIndices((SLuint)indices.size(), 
                                                                        BT_uint, 
                                                                        (void*)&indices[0]);}
        
        //! Updates a specific vertex attribute in the VBO
        void        updateAttrib        (SLVertexAttribType type, 
                                         SLint elementSize, 
                                         void* dataPointer);
        
        //! Updates a specific vertex attribute in the VBO
        void        updateAttrib        (SLVertexAttribType type, 
                                         SLVfloat& data) {updateAttrib(type, 1, (void*)&data[0]);}
        
        //! Updates a specific vertex attribute in the VBO
        void        updateAttrib        (SLVertexAttribType type, 
                                         SLVVec2f& data) {updateAttrib(type, 2, (void*)&data[0]);}

        //! Updates a specific vertex attribute in the VBO
        void        updateAttrib        (SLVertexAttribType type, 
                                         SLVVec3f& data) {updateAttrib(type, 3, (void*)&data[0]);}

        //! Updates a specific vertex attribute in the VBO
        void        updateAttrib        (SLVertexAttribType type, 
                                         SLVVec4f& data) {updateAttrib(type, 4, (void*)&data[0]);}
        
        //! Generates the VA & VB objects for a NO. of vertices
        void        generate            (SLuint numVertices, 
                                         SLBufferUsage usage = BU_static,
                                         SLbool outputInterleaved = true);

        //! Draws the VAO by element indices with a primitive type
        void        drawElementsAs      (SLPrimitiveType primitiveType,
                                         SLuint numIndexes = 0,
                                         SLuint indexOffsetBytes = 0);
        
        //! Draws the VAO as an array with a primitive type 
        void        drawArrayAs         (SLPrimitiveType primitiveType,
                                         SLint firstVertex = 0,
                                         SLsizei countVertices = 0);

        // Some getters
        SLint       numVertices         () {return _numVertices;}
        SLint       numIndices          () {return _numIndices;}

        // Some statistics
        static SLuint totalBufferCount;     //! static total no. of buffers in use
        static SLuint totalBufferSize;      //! static total size of all buffers in bytes
        static SLuint totalDrawCalls;       //! static total no. of draw calls
                                               
    protected:
        SLbool          _glHasVAO;          //! VAOs are present if OpenGL > 3.0    
        SLVVertexAttrib _attribs;           //! Vector of vertex attributes
        SLbool          _outputInterleaved; //! Flag if VBO should be generated interleaved
        SLint           _strideBytes;       //! Distance for interleaved attributes in bytes
         
        SLuint          _idVAO;             //! OpenGL id of vertex array object
        SLuint          _idVBOAttribs;      //! OpenGL id of vertex buffer object
        SLuint          _idVBOIndices;      //! OpenGL id of index vbo
        
        SLuint          _numVertices;       //! NO. of vertices in array
        SLuint          _vboSize;           //! Total size of VBO in bytes
        SLuint          _numIndices;        //! NO. of vertex indices in array
        void*           _indexData;         //! pointer to index data
        SLBufferType    _indexDataType;     //! index data type (ubyte, ushort, uint)
        SLint           _indexTypeSize;     //! index data type size
        SLBufferUsage   _usage;             //! buffer usage (static, dynamic or stream)
};
//-----------------------------------------------------------------------------

#endif
