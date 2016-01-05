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
#include <SLGLBuffer.h>

//-----------------------------------------------------------------------------
//! Enumeration for float vertex attribute types
typedef enum
{   SL_POSITION,    //! Vertex position as a 2, 3 or 4 component vectors
    SL_NORMAL,      //! Vertex normal as a 3 component vector
    SL_TEXCOORD,    //! Vertex texture coordinate as 2 component vector
    SL_TANGENT,     //! Vertex tangent as a 4 component vector (see SLMesh) 
    SL_JOINTWEIGHT, //! Vertex joint weight for vertex skinning
    SL_JOINTINDEX,  //! Vertex joint id for vertex skinning
    SL_COLOR        //! Vertex color as 3 or 4 component vector
} SLVertexAttribType;
//-----------------------------------------------------------------------------
//! Struct for vertex attribute information
struct SLVertexAttrib
{   SLVertexAttribType type;    //! type of vertex attribute
    SLint elementSize;          //! size of attribute element (SLVec3f has 3)
    SLuint bufferOffsetBytes;   //! offset of the attribute data in the buffer
    SLuint bufferSizeBytes;     //! size of the attribute part in the buffer
    void* dataPointer;          //! pointer to the attributes source data
    SLint location;             //! GLSL input variable location index
};
//-----------------------------------------------------------------------------
typedef vector<SLVertexAttrib>  SLVVertexAttrib;
//-----------------------------------------------------------------------------
//! Encapsulation of an OpenGL Vertex Array Object (VAO)
/*! 

*/
class SLGLVertexArray
{
    public:
                    SLGLVertexArray     ();
                   ~SLGLVertexArray     ();
        
        //! Deletes all vertex array & vertex buffer objects
        void        dispose             ();

        //! Clears the attribute definition
        void        clearAttribs        () {dispose(); _attribs.clear();}

        //! Returns either the VAO id or the VBO id
        SLint       id                  () {return _glHasVAO?_idVAO:_idVBOAttribs;}

        //! Adds a vertex attribute
        void        addAttrib           (SLVertexAttribType aType, 
                                         SLint elementSize, 
                                         SLint location, 
                                         void* dataPointer);
        
        //! Adds the index array for indexed element drawing
        void        addIndices          (SLuint numIndices,
                                         SLBufferType indexDataType,
                                         void* dataPointer);
        
        //! Updates a specific vertex attribute in the VBO
        void        updateAttrib        (SLVertexAttribType aType, 
                                         SLint elementSize, 
                                         void* dataPointer);
        
        //! Generates the VA & VB objects for a NO. of vertices
        void        generate            (SLuint numVertices, 
                                         SLBufferUsage usage = SL_STATIC_DRAW);

        //! Draws the VAO by element indices with a primitive type
        void        drawElementsAs      (SLPrimitive primitiveType,
                                         SLuint numIndexes = 0,
                                         SLuint indexOffsetBytes = 0);
        
        //! Draws the VAO as an array with a primitive type 
        void        drawArrayAs         (SLPrimitive primitiveType,
                                         SLint firstVertex = 0,
                                         SLsizei countVertices = 0);

        //! Adds or updates & generates a position vertex attribute for const color line drawing
        void        generatePosAttrib   (SLuint numVertices,
                                         SLint elementSize,
                                         void* dataPointer);

        //! Draws a position vertex array as lines with const color attribute
        void        drawColorLines      (SLCol3f color,
                                         SLfloat lineSize = 1.0f,
                                         SLuint  indexFirstVertex = 0,
                                         SLuint  numVertices = 0);
                                         
        //! Draws a position vertex array as points with const color attribute
        void        drawColorPoints     (SLCol4f color,
                                         SLfloat pointSize = 1.0f,
                                         SLuint  indexFirstVertex = 0,
                                         SLuint  numVertices = 0);

        //! Returns the vector index if a vertex attribute exists otherwise -1
        SLint      attribIndex          (SLVertexAttribType aType);
      
        // Some statistics
        static SLuint totalBufferCount; //! static total no. of buffers in use
        static SLuint totalBufferSize;  //! static total size of all buffers in bytes
        static SLuint totalDrawCalls;   //! static total no. of draw calls
                                               
    private:
        SLbool          _glHasVAO;      //! VAOs are present if OpenGL > 3.0    
        SLVVertexAttrib _attribs;       //! Vector of vertex attributes      
        SLuint          _idVAO;         //! OpenGL id of vertex array object
        SLuint          _idVBOAttribs;  //! OpenGL id of vertex buffer object
        SLuint          _idVBOIndices;  //! OpenGL id of index vbo
        SLuint          _numVertices;   //! NO. of vertices in array
        SLuint          _vboSize;       //! Total size of VBO in bytes
        SLuint          _numIndices;    //! NO. of vertex indices in array
        void*           _indexData;     //! pointer to index data
        SLBufferType    _indexDataType; //! index data type (ubyte, ushort, uint)
        SLint           _indexTypeSize; //! index data type size
        SLBufferUsage   _usage;         //! buffer usage (static, dynamic or stream)
};
//-----------------------------------------------------------------------------

#endif
