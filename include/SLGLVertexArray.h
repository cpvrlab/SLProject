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
typedef enum
{   SL_FLOAT          = GL_FLOAT,          // vertex attributes (position, normals)
    SL_UNSIGNED_BYTE  = GL_UNSIGNED_BYTE,  // vertex index type (0-2^8)
    SL_UNSIGNED_SHORT = GL_UNSIGNED_SHORT, // vertex index type (0-2^16)
    SL_UNSIGNED_INT   = GL_UNSIGNED_INT    // vertex index type (0-2^32)
} SLBufferType;
//-----------------------------------------------------------------------------
//! Enumeration for buffer target types
typedef enum
{   SL_ARRAY_BUFFER         = GL_ARRAY_BUFFER,         // vertex attributes arrays
    SL_ELEMENT_ARRAY_BUFFER = GL_ELEMENT_ARRAY_BUFFER  // vertex index arrays
} SLBufferTarget;
//-----------------------------------------------------------------------------
/*! Enumeration for buffer usage types also supported by OpenGL ES
STATIC:  Buffer contents will be modified once and used many times.
STREAM:  Buffer contents will be modified once and used at most a few times.
DYNAMIC: Buffer contents will be modified repeatedly and used many times.
*/
typedef enum
{   SL_STATIC_DRAW  = GL_STATIC_DRAW,
    SL_STREAM_DRAW  = GL_STREAM_DRAW,
    SL_DYNAMIC_DRAW = GL_DYNAMIC_DRAW,
} SLBufferUsage;
//-----------------------------------------------------------------------------
// Enumeration for OpenGL primitive types also available on OpenGL ES
typedef enum
{   SL_POINTS         = GL_POINTS,
    SL_LINES          = GL_LINES,
    SL_LINE_LOOP      = GL_LINE_LOOP,
    SL_LINE_STRIP     = GL_LINE_STRIP,
    SL_TRIANGLES      = GL_TRIANGLES,
    SL_TRIANGLE_STRIP = GL_TRIANGLE_STRIP,
    SL_TRIANGLE_FAN   = GL_TRIANGLE_FAN
} SLPrimitive;
//-----------------------------------------------------------------------------
//! Enumeration for float vertex attribute types
typedef enum
{   SL_POSITION,    //! Vertex position as a 2, 3 or 4 component vectors
    SL_NORMAL,      //! Vertex normal as a 3 component vector
    SL_TEXCOORD,    //! Vertex texture coordinate as 2 component vector
    SL_TANGENT,     //! Vertex tangent as a 4 component vector (see SLMesh) 
    SL_JOINTWEIGHT, //! Vertex joint weight for vertex skinning
    SL_JOINTINDEX,  //! Vertex joint id for vertex skinning
    SL_COLOR,       //! Vertex color as 3 or 4 component vector
    SL_CUSTOM1,     //! Custom vertex attribute 1
    SL_CUSTOM2,     //! Custom vertex attribute 2
    SL_CUSTOM3,     //! Custom vertex attribute 3
    SL_CUSTOM4,     //! Custom vertex attribute 4
    SL_CUSTOM5      //! Custom vertex attribute 5
} SLVertexAttribType;
//-----------------------------------------------------------------------------
//! Struct for vertex attribute information
struct SLVertexAttrib
{   SLVertexAttribType type;    //! type of vertex attribute
    SLint elementSize;          //! size of attribute element (SLVec3f has 3)
    SLuint offsetBytes;         //! offset of the attribute data in the buffer
    SLuint bufferSizeBytes;     //! size of the attribute part in the buffer
    void* dataPointer;          //! pointer to the attributes source data
    SLint location;             //! GLSL input variable location index
};
//-----------------------------------------------------------------------------
typedef vector<SLVertexAttrib>  SLVVertexAttrib;




//-----------------------------------------------------------------------------
//! SLGLVertexArray encapsulates the core OpenGL drawing
/*! An SLGLVertexArray instance handles all OpenGL drawing with an OpenGL 
Vertex Array Object (VAO )(if available) and a Vertex Buffer Object (VBO). 
VAOs where introduces OpenGL 3.0 and reduce the per draw call overhead. 
All vertex attributes (e.g. position, normals, texture coords, etc.) are float
and are stored in one big VBO. They can be in sequential order (first all 
positions, then all normals, etc.) or interleaved (all attributes together for
one vertex).\n
Vertices can be drawn either directly as in the array (SLGLVertexArray::drawArrayAs) 
or by element (SLGLVertexArray::drawElementAs) with a separate indices VBO.\n
The setup of a VAO has multiple steps:\n
- Define one ore more attributes with SLGLVertexArray::setAttrib.
- Define the index array for element drawing with SLGLVertexArray::setIndices.
- Generate the OpenGL VAO and VBO with SLGLVertexArray::generate.
It is important that the data structures passed in setAttrib and setIndices
are still present when generate is called.
*/
class SLGLVertexArray
{
    public:
                    SLGLVertexArray     ();
                   ~SLGLVertexArray     ();
        
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
        void        setIndices          (SLVubyte indices) {setIndices((SLuint)indices.size(), 
                                                                       SL_UNSIGNED_BYTE, 
                                                                       (void*)&indices[0]);}
        
        //! Adds the index array for indexed element drawing with a vector of ushort
        void        setIndices          (SLVushort indices) {setIndices((SLuint)indices.size(), 
                                                                        SL_UNSIGNED_SHORT, 
                                                                        (void*)&indices[0]);}
        
        //! Adds the index array for indexed element drawing with a vector of uint
        void        setIndices          (SLVuint indices) {setIndices((SLuint)indices.size(), 
                                                                      SL_UNSIGNED_INT, 
                                                                      (void*)&indices[0]);}
        
        //! Updates a specific vertex attribute in the VBO
        void        updateAttrib        (SLVertexAttribType type, 
                                         SLint elementSize, 
                                         void* dataPointer);
        
        //! Generates the VA & VB objects for a NO. of vertices
        void        generate            (SLuint numVertices, 
                                         SLBufferUsage usage = SL_STATIC_DRAW,
                                         SLbool outputInterleaved = true);

        //! Draws the VAO by element indices with a primitive type
        void        drawElementsAs      (SLPrimitive primitiveType,
                                         SLuint numIndexes = 0,
                                         SLuint indexOffsetBytes = 0);
        
        //! Draws the VAO as an array with a primitive type 
        void        drawArrayAs         (SLPrimitive primitiveType,
                                         SLint firstVertex = 0,
                                         SLsizei countVertices = 0);

        //////////////////////////////////////////////////
        // Helper Functions for quick Line & Point Drawing 
        //////////////////////////////////////////////////

        //! Adds or updates & generates a position vertex attribute for colored line or point drawing
        void        generateVertexPos   (SLuint numVertices,
                                         SLint elementSize,
                                         void* dataPointer);

        //! Adds or updates & generates a position vertex attribute for colored line or point drawing
        void        generateVertexPos   (SLVVec2f p) {generateVertexPos((SLuint)p.size(), 2, (void*)&p[0]);}

        //! Adds or updates & generates a position vertex attribute for colored line or point drawing
        void        generateVertexPos   (SLVVec3f p) {generateVertexPos((SLuint)p.size(), 3, (void*)&p[0]);}

        //! Adds or updates & generates a position vertex attribute for colored line or point drawing
        void        generateVertexPos   (SLVVec4f p) {generateVertexPos((SLuint)p.size(), 4, (void*)&p[0]);}
        
        //! Draws the array as the specified primitive with the color 
        void        drawArrayAsColored  (SLPrimitive primitiveType,
                                         SLCol4f color,
                                         SLfloat lineOrPointSize = 1.0f,
                                         SLuint  indexFirstVertex = 0,
                                         SLuint  countVertices = 0);
         

        // Some getter
        SLint       numVertices         () {return _numVertices;}
        SLint       numIndices          () {return _numIndices;}

        // Some statistics
        static SLuint totalBufferCount;     //! static total no. of buffers in use
        static SLuint totalBufferSize;      //! static total size of all buffers in bytes
        static SLuint totalDrawCalls;       //! static total no. of draw calls
                                               
    private:
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
