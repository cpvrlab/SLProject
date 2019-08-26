//#############################################################################
//  File:      SLGLVertexArray.cpp
//  Purpose:   Wrapper around an OpenGL Vertex Array Objects
//  Author:    Marcus Hudritsch
//  Date:      January 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SLApplication.h>
#include <SLGLProgram.h>
#include <SLGLVertexArray.h>
#include <SLScene.h>

//-----------------------------------------------------------------------------
SLuint SLGLVertexArray::totalDrawCalls = 0;
//-----------------------------------------------------------------------------
/*! Constructor initializing with default values
*/
SLGLVertexArray::SLGLVertexArray()
{
    _hasGL3orGreater = SLGLState::instance()->glVersionNOf() >= 3.0f;
    _idVAO           = 0;

    _VBOf.dataType(BT_float);
    _VBOf.clear();
    _idVBOIndices = 0;
    _numIndices   = 0;
    _numVertices  = 0;
}
//-----------------------------------------------------------------------------
/*! Deletes the OpenGL objects for the vertex array and the vertex buffer.
The vector _attribs with the attribute information is not cleared.
*/
void SLGLVertexArray::deleteGL()
{
#ifndef APP_USES_GLES
    if (_hasGL3orGreater && _idVAO)
    {
        glDeleteVertexArrays(1, &_idVAO);
    }
#endif
    _idVAO = 0;

    if (_VBOf.id()) _VBOf.clear();

    if (_idVBOIndices)
    {
        glDeleteBuffers(1, &_idVBOIndices);
        _idVBOIndices = 0;
        SLGLVertexBuffer::totalBufferCount--;
        SLGLVertexBuffer::totalBufferSize -= _numIndices * (SLuint)SLGLVertexBuffer::sizeOfType(_indexDataType);
    }
}
//-----------------------------------------------------------------------------
// Returns the vertex array object id
SLint SLGLVertexArray::id()
{
#ifndef APP_USES_GLES
    return _hasGL3orGreater ? (SLint)_idVAO : (SLint)_VBOf.id();
#else
    return _VBOf.id();
#endif
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
                                SLint             elementSize,
                                SLint             location,
                                void*             dataPointer)
{
    assert(dataPointer);
    assert(elementSize);

    if (type == AT_position && location == -1)
        SL_EXIT_MSG("The position attribute has no variable location.");

    if (_VBOf.attribIndex(type) >= 0)
        SL_EXIT_MSG("Attribute type already exists.");

    SLGLAttribute va;
    va.type            = type;
    va.elementSize     = elementSize;
    va.dataPointer     = dataPointer;
    va.location        = location;
    va.bufferSizeBytes = 0;

    _VBOf.attribs().push_back(va);
}
//-----------------------------------------------------------------------------
/*! Defines the vertex indices for the element drawing. Without indices vertex
array can only be drawn with SLGLVertexArray::drawArrayAs. 
Be aware that the VBO for the indices will not be generated until generate 
is called. The data pointer must still be valid when generate is called. 
*/
void SLGLVertexArray::setIndices(SLuint         numIndices,
                                 SLGLBufferType indexDataType,
                                 void*          dataPointer)
{
    assert(numIndices);
    assert(dataPointer);

    if (indexDataType == BT_ushort && _numVertices > 65535)
        SL_EXIT_MSG("Index data type not sufficient.");
    if (indexDataType == BT_ubyte && _numVertices > 255)
        SL_EXIT_MSG("Index data type not sufficient.");

    _numIndices    = numIndices;
    _indexDataType = indexDataType;
    _indexData     = dataPointer;
}
//-----------------------------------------------------------------------------
/*! Updates the specified vertex attribute. This works only for sequential 
attributes and not for interleaved attributes. This is used e.g. for meshes
with vertex skinning. See SLMesh::draw where we have joint attributes.
*/
void SLGLVertexArray::updateAttrib(SLGLAttributeType type,
                                   SLint             elementSize,
                                   void*             dataPointer)
{
    assert(dataPointer && "No data pointer passed");
    assert(elementSize > 0 && elementSize < 5 && "Element size invalid");

    // Get attribute index and check element size
    SLint indexf = _VBOf.attribIndex(type);
    if (indexf == -1)
        SL_EXIT_MSG("Attribute type does not exist in VAO.");

#ifndef APP_USES_GLES
    if (_hasGL3orGreater)
    {
        if (!_idVAO)
            glGenVertexArrays(1, &_idVAO);
        glBindVertexArray(_idVAO);
    }
#endif

    // update the appropriate VBO
    if (indexf > -1)
        _VBOf.updateAttrib(type, elementSize, dataPointer);

#ifndef APP_USES_GLES
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
void SLGLVertexArray::generate(SLuint          numVertices,
                               SLGLBufferUsage usage,
                               SLbool          outputinterleaved)
{
    assert(numVertices);

    // if buffers exist delete them first
    deleteGL();

    _numVertices = numVertices;

// Generate and bind VAO
#ifndef APP_USES_GLES
    if (_hasGL3orGreater)
    {
        glGenVertexArrays(1, &_idVAO);
        glBindVertexArray(_idVAO);
    }
#endif

    ///////////////////////////////
    // Create Vertex Buffer Objects
    ///////////////////////////////

    // Generate the vertex buffer object for float attributes
    if (_VBOf.attribs().size())
        _VBOf.generate(numVertices, usage, outputinterleaved);

    //////////////////////////////////////////
    // Create Element Array Buffer for Indices
    //////////////////////////////////////////

    if (_numIndices)
    {
        SLuint typeSize = SLGLVertexBuffer::sizeOfType(_indexDataType);
        glGenBuffers(1, &_idVBOIndices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idVBOIndices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     _numIndices * (SLuint)typeSize,
                     _indexData,
                     GL_STATIC_DRAW);
        SLGLVertexBuffer::totalBufferCount++;
        SLGLVertexBuffer::totalBufferSize += _numIndices * (SLuint)typeSize;
    }

#ifndef APP_USES_GLES
    if (_hasGL3orGreater)
        glBindVertexArray(0);
#endif

#ifdef _GLDEBUG
    GET_GL_ERROR;
#endif
}
//-----------------------------------------------------------------------------
/*! Draws the vertex attributes as a specified primitive type by elements with 
the indices from the index buffer defined in setIndices.
*/
void SLGLVertexArray::drawElementsAs(SLGLPrimitiveType primitiveType,
                                     SLuint            numIndexes,
                                     SLuint            indexOffset)
{
    assert(_numIndices && _idVBOIndices && "No index VBO generated for VAO");

    // From OpenGL 3.0 on we have the OpenGL Vertex Arrays
    // Binding the VAO saves all the commands after the else (per draw call!)

#ifndef APP_USES_GLES
    if (_hasGL3orGreater)
    {
        glBindVertexArray(_idVAO);
        GET_GL_ERROR;
    }
#else
    if (!_VBOf.id())
        SL_EXIT_MSG("No VBO generated for VAO.");
    _VBOf.bindAndEnableAttrib();

    // Activate the index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idVBOIndices);
    GET_GL_ERROR;
#endif

    // Do the draw call with indices
    if (numIndexes == 0)
        numIndexes = _numIndices;

    SLuint indexTypeSize = SLGLVertexBuffer::sizeOfType(_indexDataType);

    ///////////////////////////////////////////////////////////////////
    glDrawElements(primitiveType,
                   (SLsizei)numIndexes,
                   _indexDataType,
                   (void*)(size_t)(indexOffset * (SLuint)indexTypeSize));
    ///////////////////////////////////////////////////////////////////

    GET_GL_ERROR;
    totalDrawCalls++;

#ifndef APP_USES_GLES
    if (_hasGL3orGreater)
        glBindVertexArray(0);
#else
    _VBOf.disableAttrib();
#endif

#ifdef _GLDEBUG
    GET_GL_ERROR;
#endif
}
//-----------------------------------------------------------------------------
/*! Draws the vertex attributes as a specified primitive type as the vertices
are defined in the attribute arrays.
*/
void SLGLVertexArray::drawArrayAs(SLGLPrimitiveType primitiveType,
                                  SLint             firstVertex,
                                  SLsizei           countVertices)
{
    assert((_VBOf.id()) && "No VBO generated for VAO.");

#ifndef APP_USES_GLES
    if (_hasGL3orGreater)
        glBindVertexArray(_idVAO);
#else
    _VBOf.bindAndEnableAttrib();
#endif

    if (countVertices == 0)
        countVertices = (SLsizei)_numVertices;

    ////////////////////////////////////////////////////////
    glDrawArrays(primitiveType, firstVertex, countVertices);
    ////////////////////////////////////////////////////////

    totalDrawCalls++;

#ifndef APP_USES_GLES
    if (_hasGL3orGreater)
        glBindVertexArray(0);
#else
    _VBOf.disableAttrib();
#endif

#ifdef _GLDEBUG
    GET_GL_ERROR;
#endif
}
//-----------------------------------------------------------------------------
