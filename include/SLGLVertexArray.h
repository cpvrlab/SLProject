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
#include <SLGLEnums.h>
#include <SLGLVertexBuffer.h>

//-----------------------------------------------------------------------------
//! SLGLVertexArray encapsulates the core OpenGL drawing
/*! An SLGLVertexArray instance handles all OpenGL drawing with an OpenGL 
Vertex Array Object (VAO), a vertex buffer objects (VBO) for the attributes
and an index buffer for element drawing. Attributes can be stored in a float
or half float VBO of type SLGLVertexBuffer.\n 
VAOs where introduces OpenGL 3.0 and reduce the overhead per draw call. 
All vertex attributes (e.g. position, normals, texture coords, etc.) must be
float at the input. They can be flagged to be converted to half floats so that
they will be stored in a separate VBO. All float attributes will be in one VBO
(_VBOf) and all half float attributes are stored in one half float VBO (_VBOh).
Half float attributes only use 2 bytes per number but will be converted to 4
byte floats before they arrive in the shader. 
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
    public:         SLGLVertexArray     ();
                   ~SLGLVertexArray     () {deleteGL();}
        
        //! Deletes all vertex array & vertex buffer objects
        void        deleteGL            ();

        //! Clears the attribute definition
        void        clearAttribs        () {deleteGL(); 
                                            _VBOf.clear(BT_float); 
                                            _VBOh.clear(BT_half);}

        //! Returns either the VAO id or the VBO id
        SLint       id                  () {return _glHasVAO?_idVAO:_VBOf.id();}

        //! Returns the vector index if a float vertex attribute exists otherwise -1
        SLint       attribIndexh        (SLGLAttributeType type);

        //! Adds a vertex attribute with data pointer and an element size
        void        setAttrib           (SLGLAttributeType type, 
                                         SLint elementSize, 
                                         SLint location, 
                                         void* dataPointer,
                                         SLbool convertToHalf=false);

        //! Adds a vertex attribute with vector of SLfloat
        void        setAttrib           (SLGLAttributeType type,
                                         SLint location, 
                                         SLVfloat& data,
                                         SLbool convertToHalf=false) {setAttrib(type, 1, location, (void*)&data[0]);}

        //! Adds a vertex attribute with vector of SLVec2f
        void        setAttrib           (SLGLAttributeType type,
                                         SLint location, 
                                         SLVVec2f& data,
                                         SLbool convertToHalf=false) {setAttrib(type, 2, location, (void*)&data[0]);}

        //! Adds a vertex attribute with vector of SLVec3f
        void        setAttrib           (SLGLAttributeType type,
                                         SLint location, 
                                         SLVVec3f& data,
                                         SLbool convertToHalf=false) {setAttrib(type, 3, location, (void*)&data[0]);}

        //! Adds a vertex attribute with vector of SLVec4f
        void        setAttrib           (SLGLAttributeType type,
                                         SLint location, 
                                         SLVVec4f& data,
                                         SLbool convertToHalf=false) {setAttrib(type, 4, location, (void*)&data[0]);}
        
        //! Adds the index array for indexed element drawing
        void        setIndices          (SLuint numIndices,
                                         SLGLBufferType indexDataType,
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
        void        updateAttrib        (SLGLAttributeType type, 
                                         SLint elementSize, 
                                         void* dataPointer);
        
        //! Updates a specific vertex attribute in the VBO
        void        updateAttrib        (SLGLAttributeType type, 
                                         SLVfloat& data) {updateAttrib(type, 1, (void*)&data[0]);}
        
        //! Updates a specific vertex attribute in the VBO
        void        updateAttrib        (SLGLAttributeType type, 
                                         SLVVec2f& data) {updateAttrib(type, 2, (void*)&data[0]);}

        //! Updates a specific vertex attribute in the VBO
        void        updateAttrib        (SLGLAttributeType type, 
                                         SLVVec3f& data) {updateAttrib(type, 3, (void*)&data[0]);}

        //! Updates a specific vertex attribute in the VBO
        void        updateAttrib        (SLGLAttributeType type, 
                                         SLVVec4f& data) {updateAttrib(type, 4, (void*)&data[0]);}
        
        //! Generates the VA & VB objects for a NO. of vertices
        void        generate            (SLuint numVertices, 
                                         SLGLBufferUsage usage = BU_static,
                                         SLbool outputInterleaved = true);

        //! Draws the VAO by element indices with a primitive type
        void        drawElementsAs      (SLGLPrimitiveType primitiveType,
                                         SLuint numIndexes = 0,
                                         SLuint indexOffsetBytes = 0);
        
        //! Draws the VAO as an array with a primitive type 
        void        drawArrayAs         (SLGLPrimitiveType primitiveType,
                                         SLint firstVertex = 0,
                                         SLsizei countVertices = 0);

        // Some getters
        SLint       numVertices         () {return _numVertices;}
        SLint       numIndices          () {return _numIndices;}

        // Some statistics
        static SLuint totalDrawCalls;       //! static total no. of draw calls
            
                                               
    protected:
        SLbool              _glHasVAO;          //! VAOs are present if OpenGL > 3.0   
        SLuint              _idVAO;             //! OpenGL id of vertex array object
        SLuint              _numVertices;       //! NO. of vertices in array
        SLGLVertexBuffer    _VBOf;              //! Vertex buffer object for float attributes 
        SLGLVertexBuffer    _VBOh;              //! Vertex buffer object for half float attributes

        SLuint              _idVBOIndices;      //! OpenGL id of index vbo
        SLuint              _numIndices;        //! NO. of vertex indices in array
        void*               _indexData;         //! pointer to index data
        SLGLBufferType      _indexDataType;     //! index data type (ubyte, ushort, uint)
};
//-----------------------------------------------------------------------------

#endif
