//#############################################################################
//  File:      SLGLVertexArray.cpp
//  Purpose:   Wrapper around an OpenGL Vertex Array Objects 
//  Author:    Marcus Hudritsch
//  Date:      January 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include <SLGLVertexArray.h>
#include <SLGLProgram.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
SLuint SLGLVertexArray::totalDrawCalls   = 0;
//-----------------------------------------------------------------------------
/*! Constructor initializing with default values
*/
SLGLVertexArray::SLGLVertexArray()
{   
    _hasGL3orGreater = SLGLState::getInstance()->glVersionNOf() >= 3.0f;
    _idVAO = 0;
    _VBOf.dataType(BT_float);
    _VBOh.dataType(BT_half);
    _VBOf.clear();
    _VBOh.clear();
    _idVBOIndices = 0;
    _numIndices = 0;
    _numVertices = 0;
}
//-----------------------------------------------------------------------------
/*! Deletes the OpenGL objects for the vertex array and the vertex buffer.
The vector _attribs with the attribute information is not cleared.
*/
void SLGLVertexArray::deleteGL()
{
    if (_hasGL3orGreater && _idVAO)
    {   glDeleteVertexArrays(1, &_idVAO);
        _idVAO = 0;
    }
    
    if (_VBOf.id()) _VBOf.clear();
    if (_VBOh.id()) _VBOh.clear();

    if (_idVBOIndices)
    {   glDeleteBuffers(1, &_idVBOIndices);
        _idVBOIndices = 0;
        SLGLVertexBuffer::totalBufferCount--;
        SLGLVertexBuffer::totalBufferSize -= _numIndices * SLGLVertexBuffer::sizeOfType(_indexDataType);
    }
}

//-----------------------------------------------------------------------------
/*! Defines a vertex attribute for the later generation. 
It must be of a specific SLVertexAttribType. Each attribute can appear only 
once in an vertex array.
If all attributes of a vertex array have the same data pointer the data input 
will be interpreted as an interleaved array. See example in SLGLOculus::init.
Be aware that the VBO for the attribute will not be generated until generate 
is called. The data pointer must still be valid when SLGLVertexArray::generate 
is called.
*/
void SLGLVertexArray::setAttrib(SLGLAttributeType type, 
                                SLint elementSize,
                                SLint location, 
                                void* dataPointer,
                                SLbool convertToHalf)
{   assert(dataPointer);
    assert(elementSize);

    if (type == AT_position && location == -1)
        SL_EXIT_MSG("The position attribute has no variable location.");

    if (type == AT_position && convertToHalf)
        SL_EXIT_MSG("The position attribute should be from float data type.");

    if (_VBOf.attribIndex(type) >= 0 || _VBOh.attribIndex(type) >= 0)
        SL_EXIT_MSG("Attribute type already exists.");

    SLGLAttribute va;
    va.type = type;
    va.elementSize = elementSize;
    va.dataPointer = dataPointer;
    va.location = location;
    va.bufferSizeBytes = 0;

    if (convertToHalf && _hasGL3orGreater)
         _VBOh.attribs().push_back(va);
    else _VBOf.attribs().push_back(va);
}
//-----------------------------------------------------------------------------
/*! Defines the vertex indices for the element drawing. Without indices vertex
array can only be drawn with SLGLVertexArray::drawArrayAs. 
Be aware that the VBO for the indices will not be generated until generate 
is called. The data pointer must still be valid when generate is called. 
*/
void SLGLVertexArray::setIndices(SLuint numIndices,
                                 SLGLBufferType indexDataType,
                                 void* dataPointer)
{   assert(numIndices);
    assert(dataPointer);
    
    if (indexDataType == BT_ushort && numIndices > 65535)
        SL_EXIT_MSG("Index data type not sufficient.");
    if (indexDataType == BT_ubyte && numIndices > 255)
        SL_EXIT_MSG("Index data type not sufficient.");
        
    _numIndices = numIndices;
    _indexDataType = indexDataType;
    _indexData = dataPointer;
}
//-----------------------------------------------------------------------------
/*! Updates the specified vertex attribute. This works only for sequential 
attributes and not for interleaved attributes. This is used e.g. for meshes
with vertex skinning. See SLMesh::draw where we have joint attributes.
*/
void SLGLVertexArray::updateAttrib(SLGLAttributeType type, 
                                   SLint elementSize,
                                   void* dataPointer)
{   
    assert(dataPointer && "No data pointer passed");
    assert(elementSize > 0 && elementSize < 5 && "Element size invalid");
    
    // Get attribute index and check element size
    SLint indexf = _VBOf.attribIndex(type);
    SLint indexh = _VBOh.attribIndex(type);
    if (indexf == -1 && indexh == -1)
        SL_EXIT_MSG("Attribute type does not exist in VAO.");
    
    #ifndef SL_GLES2
    if (_hasGL3orGreater)
    {   if (!_idVAO)
            glGenVertexArrays(1, &_idVAO);
        glBindVertexArray(_idVAO);
    }
    #endif

    // update the appropriate VBO
    if (indexf>-1) 
        _VBOf.updateAttrib(type, elementSize, dataPointer);
    if (indexh>-1) 
        _VBOh.updateAttrib(type, elementSize, dataPointer);

    #ifndef SL_GLES2
    if (_hasGL3orGreater)
        glBindVertexArray(0);
    #endif
    
    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
/*! Generates the OpenGL objects for the vertex array (if available) and the 
vertex buffer object. If the input data is an interleaved array (all attribute
data pointer where identical) also the output buffer will be generated as an
interleaved array. Vertex arrays with attributes that are updated can not be
interleaved. Vertex attributes with separate arrays can generate an interleaved
or a sequential vertex buffer.\n\n
<PRE>
\n Sequential attribute layout:                                                          
\n           |          Positions          |           Normals           |     TexCoords     |   
\n Attribs:  |   Position0  |   Position1  |    Normal0   |    Normal1   |TexCoord0|TexCoord1|   
\n Elements: | PX | PY | PZ | PX | PY | PZ | NX | NY | NZ | NX | NY | NZ | TX | TY | TX | TY |   
\n Bytes:    |#### #### ####|#### #### ####|#### #### ####|#### #### ####|#### ####|#### ####|  
\n           |                             |                             |
\n           |<------ offset Normals ----->|                             |
\n           |<-------------------- offset TexCoords ------------------->|
\n                                                                                               
\n Interleaved attribute layout:                                                                
\n           |               Vertex 0                |               Vertex 1                |   
\n Attribs:  |   Position0  |    Normal0   |TexCoord0|   Position1  |    Normal1   |TexCoord1|   
\n Elements: | PX | PY | PZ | NX | NY | NZ | TX | TY | PX | PY | PZ | NX | NY | NZ | TX | TY |   
\n Bytes:    |#### #### ####|#### #### ####|#### ####|#### #### ####|#### #### ####|#### ####|    
\n           |              |              |         |
\n           |<-offsetN=32->|              |         |
\n           |<------- offsetTC=32 ------->|         |
\n           |                                       |                                            
\n           |<---------- strideBytes=32 ----------->|
</PRE>
*/
void SLGLVertexArray::generate(SLuint numVertices, 
                               SLGLBufferUsage usage,
                               SLbool outputinterleaved)
{   assert(numVertices);

    // if buffers exist delete them first
    deleteGL();

    _numVertices = numVertices;

    // Generate and bind VAO
    if (_hasGL3orGreater)
    {   glGenVertexArrays(1, &_idVAO);
        glBindVertexArray(_idVAO);
    }
    
    
    ///////////////////////////////
    // Create Vertex Buffer Objects
    ///////////////////////////////

    // Generate the vertex buffer object for float attributes
    if (_VBOf.attribs().size())
        _VBOf.generate(numVertices, usage, outputinterleaved);
        
    // Generate the vertex buffer object for half float attributes
    if (_VBOh.attribs().size())
        _VBOh.generate(numVertices, usage, outputinterleaved);


    //////////////////////////////////////////
    // Create Element Array Buffer for Indices
    //////////////////////////////////////////

    if (_numIndices)
    {   
        SLint typeSize = SLGLVertexBuffer::sizeOfType(_indexDataType);
        glGenBuffers(1, &_idVBOIndices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idVBOIndices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
                     _numIndices * typeSize, 
                     _indexData, 
                     GL_STATIC_DRAW);
        SLGLVertexBuffer::totalBufferCount++;
        SLGLVertexBuffer::totalBufferSize += _numIndices * typeSize;
    }

    if (_hasGL3orGreater)
        glBindVertexArray(0);
    
    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
/*! Draws the vertex attributes as a specified primitive type by elements with 
the indices from the index buffer defined in setIndices.
*/
void SLGLVertexArray::drawElementsAs(SLGLPrimitiveType primitiveType,
                                     SLuint numIndexes,
                                     SLuint indexOffset)
{   
    assert((_VBOf.id() || _VBOh.id()) && "No VBO generated for VAO.");
    assert(_numIndices && _idVBOIndices && "No index VBO generated for VAO");

    // From OpenGL 3.0 on we have the OpenGL Vertex Arrays
    // Binding the VAO saves all the commands after the else (per draw call!)
    
    if (_hasGL3orGreater)
    {   glBindVertexArray(_idVAO);
        GET_GL_ERROR;
    }
    else
    {   _VBOf.bindAndEnableAttrib();
        _VBOh.bindAndEnableAttrib();

        // Activate the index buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idVBOIndices);
        GET_GL_ERROR;
    }

    // Do the draw call with indices 
    if (numIndexes==0) 
        numIndexes = _numIndices;
    
    SLint indexTypeSize = SLGLVertexBuffer::sizeOfType(_indexDataType);

    ////////////////////////////////////////////////////////////
    glDrawElements(primitiveType, 
                   numIndexes, 
                   _indexDataType, 
                   (void*)(size_t)(indexOffset*indexTypeSize));
    ////////////////////////////////////////////////////////////
    
    GET_GL_ERROR;
    totalDrawCalls++;

    if (_hasGL3orGreater)
        glBindVertexArray(0);
    else
    {   _VBOf.disableAttrib();
        _VBOh.disableAttrib();
    }

    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
/*! Draws the vertex attributes as a specified primitive type as the vertices
are defined in the attribute arrays.
*/
void SLGLVertexArray::drawArrayAs(SLGLPrimitiveType primitiveType,
                                  SLint firstVertex,
                                  SLsizei countVertices)
{   
    assert((_VBOf.id() || _VBOh.id()) && "No VBO generated for VAO.");
    
    if (_hasGL3orGreater)
        glBindVertexArray(_idVAO);
    else
    {   _VBOf.bindAndEnableAttrib();
        _VBOh.bindAndEnableAttrib();
    }

    if (countVertices == 0)
        countVertices = _numVertices;

    ////////////////////////////////////////////////////////
    glDrawArrays(primitiveType, firstVertex, countVertices);
    ////////////////////////////////////////////////////////
    
    totalDrawCalls++;
    
    if (_hasGL3orGreater)
        glBindVertexArray(0);
    else
    {   _VBOf.disableAttrib();
        _VBOh.disableAttrib();
    }

    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
