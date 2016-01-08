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
SLuint SLGLVertexArray::totalBufferSize  = 0;
SLuint SLGLVertexArray::totalBufferCount = 0;
SLuint SLGLVertexArray::totalDrawCalls   = 0;
//-----------------------------------------------------------------------------
/*! Constructor initializing with default values
*/
SLGLVertexArray::SLGLVertexArray()
{   
    _glHasVAO = SLGLState::getInstance()->glVersionNOf() > 3.0f;
    _idVAO = 0;
    _idVBOAttribs = 0;
    _idVBOIndices = 0;
    _numIndices = 0;
    _numVertices = 0;
    _vboSize = 0;
    _outputInterleaved = false;
}
//-----------------------------------------------------------------------------
/*! Deletes the OpenGL objects for the vertex array and the vertex buffer.
The vector _attribs with the attribute information is not cleared.
*/
void SLGLVertexArray::deleteGL()
{
    #ifndef SL_GLES2
    if (_glHasVAO && _idVAO) 
    {   glDeleteVertexArrays(1, &_idVAO);
        _idVAO = 0;
    }
    #endif
    
    if (_idVBOAttribs)
    {   glDeleteBuffers(1, &_idVBOAttribs);
        _idVBOAttribs = 0;
        totalBufferCount--;
        totalBufferSize -= _vboSize;
    }
    if (_idVBOIndices)
    {   glDeleteBuffers(1, &_idVBOIndices);
        _idVBOIndices = 0;
        totalBufferCount--;
        totalBufferSize -= _numIndices * _indexTypeSize;
    }
}
//-----------------------------------------------------------------------------
SLint SLGLVertexArray::attribIndex(SLVertexAttribType type)
{    
    for (SLint i=0; i<_attribs.size(); ++i)
        if (_attribs[i].type == type)
            return i;
    return -1;
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
void SLGLVertexArray::setAttrib(SLVertexAttribType type, 
                                SLint elementSize,
                                SLint location, 
                                void* dataPointer)
{   assert(dataPointer);
    assert(elementSize);

    if (type == SL_POSITION && location == -1)
        SL_EXIT_MSG("The position attribute has no variable location.");

    if (attribIndex(type) >= 0)
        SL_EXIT_MSG("Attribute type already exists.");

    SLVertexAttrib va;
    va.type = type;
    va.elementSize = elementSize;
    va.dataPointer = dataPointer;
    va.location = location;

    _attribs.push_back(va);
}
//-----------------------------------------------------------------------------
/*! Defines the vertex indices for the element drawing. Without indices vertex
array can only be drawn with SLGLVertexArray::drawArrayAs. 
Be aware that the VBO for the indices will not be generated until generate 
is called. The data pointer must still be valid when generate is called. 
*/
void SLGLVertexArray::setIndices(SLuint numIndices,
                                 SLBufferType indexDataType,
                                 void* dataPointer)
{   assert(numIndices);
    assert(dataPointer);

    switch (indexDataType)
    {   case GL_UNSIGNED_BYTE:  _indexTypeSize = sizeof(GLubyte);  break;
        case GL_UNSIGNED_SHORT: _indexTypeSize = sizeof(GLushort); break;
        case GL_UNSIGNED_INT:   _indexTypeSize = sizeof(GLuint);   break;
        default: SL_EXIT_MSG("Invalid index data type");
    }
    
    if (indexDataType == SL_UNSIGNED_SHORT && numIndices > 65535)
        SL_EXIT_MSG("Index data type not sufficient.");
    if (indexDataType == SL_UNSIGNED_BYTE && numIndices > 255)
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
void SLGLVertexArray::updateAttrib(SLVertexAttribType type, 
                                   SLint elementSize,
                                   void* dataPointer)
{   
    assert(dataPointer && "No data pointer passed");
    assert(elementSize > 0 && elementSize < 5 && "Element size invalid");
    assert(!_outputInterleaved && "Interleaved buffers can't be updated.");
    
    // Get attribute index and check element size
    SLint index = attribIndex(type);
    if (index == -1)
        SL_EXIT_MSG("Attribute type does not exist in VAO.");
    if (_attribs[index].elementSize != elementSize)
        SL_EXIT_MSG("Attribute element size differs.");
    
    #ifndef SL_GLES2
    if (_glHasVAO)
        glBindVertexArray(_idVAO);
    #endif
    
    // copy sub-data into existing buffer object
    glBindBuffer(GL_ARRAY_BUFFER, _idVBOAttribs);
    glBufferSubData(GL_ARRAY_BUFFER,
                    _attribs[index].offsetBytes,
                    _attribs[index].bufferSizeBytes,
                    dataPointer);

    #ifndef SL_GLES2
    if (_glHasVAO)
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
\n                                                                                               
\n Interleaved attribute layout:                                                                
\n           |               Vertex 0                |               Vertex 1                |   
\n Attribs:  |   Position0  |    Normal0   |TexCoord0|   Position1  |    Normal1   |TexCoord1|   
\n Elements: | PX | PY | PZ | NX | NY | NZ | TX | TY | PX | PY | PZ | NX | NY | NZ | TX | TY |   
\n Bytes:    |#### #### ####|#### #### ####|#### ####|#### #### ####|#### #### ####|#### ####|    
\n           |                                       |                                            
\n           |<--------- strideBytes = 32 ---------->|                                            
</PRE>
*/
void SLGLVertexArray::generate(SLuint numVertices, 
                               SLBufferUsage usage,
                               SLbool outputinterleaved)
{   assert(numVertices);

    // if buffers exist delete them first
    deleteGL();

    _numVertices = numVertices;
    _usage = usage;
    _outputInterleaved = outputinterleaved;

    // Generate and bind VAO
    #ifndef SL_GLES2
    if (_glHasVAO)
    {   glGenVertexArrays(1, &_idVAO);
        glBindVertexArray(_idVAO);
    }
    #endif
    
    // Generate the vertex buffer object
    glGenBuffers(1, &_idVBOAttribs);
    glBindBuffer(GL_ARRAY_BUFFER, _idVBOAttribs);

    // Check first if all attribute data pointer point to the same interleaved data
    SLbool inputIsInterleaved = false;
    if (_attribs.size() > 1)
    {   inputIsInterleaved = true;
        for (auto a : _attribs)
        {   if (a.dataPointer != _attribs[0].dataPointer)
            {   inputIsInterleaved = false;
                break;
            }
        }
    }

    ///////////////////////////////////////////////////////
    // Calculate total VBO size & attribute stride & offset
    ///////////////////////////////////////////////////////

    _vboSize = 0;
    _strideBytes = 0;

    if (inputIsInterleaved)
    {   _outputInterleaved = true;
        for (SLint i=0; i<_attribs.size(); ++i) 
        {   SLuint elementSizeBytes = _attribs[i].elementSize * sizeof(SLfloat);
            _attribs[i].offsetBytes = _strideBytes;
            _attribs[i].bufferSizeBytes = elementSizeBytes * _numVertices;
            _vboSize += _attribs[i].bufferSizeBytes;
            _strideBytes += elementSizeBytes;
        }  
    }
    else // input is in separate attribute data blocks
    {
        for (SLint i=0; i<_attribs.size(); ++i) 
        {   SLuint elementSizeBytes = _attribs[i].elementSize * sizeof(SLfloat);
            if (_outputInterleaved)
                 _attribs[i].offsetBytes = _strideBytes;
            else _attribs[i].offsetBytes = _vboSize;
            _attribs[i].bufferSizeBytes = elementSizeBytes * _numVertices;
            _vboSize += _attribs[i].bufferSizeBytes;
            if (_outputInterleaved) _strideBytes += elementSizeBytes;
        }
    }

    //////////////////////////////
    // Generate VBO for Attributes
    //////////////////////////////

    if (inputIsInterleaved)
    {
        for (auto a : _attribs)
        {   
            if (a.location > -1)
            {   // Sets the vertex attribute data pointer to its corresponding GLSL variable
                glVertexAttribPointer(a.location, 
                                      a.elementSize, 
                                      GL_FLOAT,
                                      GL_FALSE, 
                                      _strideBytes,
                                      (void*)a.offsetBytes);
        
                // Tell the attribute to be an array attribute instead of a state variable
                glEnableVertexAttribArray(a.location);
            }
        }

        // generate the interleaved vbo buffer on the GPU
        glBufferData(GL_ARRAY_BUFFER, _vboSize, _attribs[0].dataPointer, _usage);
    }
    else  // input is in separate attribute data block
    {    
        if (_outputInterleaved) // Copy attribute data interleaved
        {
            SLuchar* data = new SLuchar[_vboSize];

            for (auto a : _attribs)
            {   
                SLuint elementSizeBytes = a.elementSize * sizeof(SLfloat);

                // Copy attributes interleaved
                for (SLuint v = 0; v < _numVertices; ++v)
                {   SLuint iDst = v * _strideBytes + a.offsetBytes;
                    SLuint iSrc = v * elementSizeBytes;
                    for (SLuint b = 0; b < elementSizeBytes; ++b)
                        data[iDst+b] = ((SLuchar*)a.dataPointer)[iSrc+b];
                }

                if (a.location > -1)
                {   // Sets the vertex attribute data pointer to its corresponding GLSL variable
                    glVertexAttribPointer(a.location, 
                                          a.elementSize, 
                                          GL_FLOAT,
                                          GL_FALSE, 
                                          _strideBytes,
                                          (void*)a.offsetBytes);
        
                    // Tell the attribute to be an array attribute instead of a state variable
                    glEnableVertexAttribArray(a.location);

                }
            }

            // generate the interleaved vbo buffer on the GPU
            glBufferData(GL_ARRAY_BUFFER, _vboSize, data, _usage);
            delete[] data;
        } 
        else // copy attributes buffers sequentially
        {   
            // allocate the vbo buffer on the GPU
            glBufferData(GL_ARRAY_BUFFER, _vboSize, NULL, _usage);
        
            for (auto a : _attribs)
            {   if (a.location > -1)
                {
                    // Copies the attributes data at the right offset into the vbo
                    glBufferSubData(GL_ARRAY_BUFFER,
                                    a.offsetBytes,
                                    a.bufferSizeBytes,
                                    a.dataPointer);
        
                    // Sets the vertex attribute data pointer to its corresponding GLSL variable
                    glVertexAttribPointer(a.location, 
                                          a.elementSize, 
                                          GL_FLOAT,
                                          GL_FALSE, 
                                          0,
                                          (void*)a.offsetBytes);
        
                    // Tell the attribute to be an array attribute instead of a state variable
                    glEnableVertexAttribArray(a.location);
                }
            }
        }
    }

    totalBufferCount++;
    totalBufferSize += _vboSize;

    /////////////////////////
    // Create VBO for Indices
    /////////////////////////

    if (_numIndices)
    {   glGenBuffers(1, &_idVBOIndices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idVBOIndices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, _numIndices * _indexTypeSize, _indexData, _usage);
        totalBufferCount++;
        totalBufferSize += _numIndices * _indexTypeSize;
    }

    #ifndef SL_GLES2
    if (_glHasVAO)
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
void SLGLVertexArray::drawElementsAs(SLPrimitive primitiveType,
                                     SLuint numIndexes,
                                     SLuint indexOffset)
{   
    assert(_idVBOAttribs && "No VBO generated for VAO.");
    assert(_numIndices && _idVBOIndices && "No index VBO generated for VAO");

    // From OpenGL 3.0 on we have the OpenGL Vertex Arrays
    // Binding the VAO saves all the commands after the else (per draw call!)
    #ifndef SL_GLES2
    if (_glHasVAO)
        glBindVertexArray(_idVAO);
    else
    #endif
    {   glBindBuffer(GL_ARRAY_BUFFER, _idVBOAttribs);
        for (auto a : _attribs)
        {   if (a.location > -1)
            {
                // Sets the vertex attribute data pointer to its corresponding GLSL variable
                glVertexAttribPointer(a.location, 
                                      a.elementSize,
                                      GL_FLOAT, 
                                      GL_FALSE, 
                                      _strideBytes, 
                                      (void*)a.offsetBytes);

                // Tell the attribute to be an array attribute instead of a state variable
                glEnableVertexAttribArray(a.location);
            }
        }

        // Activate the index buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idVBOIndices);
    }

    // Do the draw call with indices 
    if (numIndexes==0) 
        numIndexes = _numIndices;

    ////////////////////////////////////////////////////
    glDrawElements(primitiveType, 
                   numIndexes, 
                   _indexDataType, 
                   (void*)(indexOffset*_indexTypeSize));
    ////////////////////////////////////////////////////
    
    totalDrawCalls++;

    #ifndef SL_GLES2
    if (_glHasVAO)
        glBindVertexArray(0);
    else
    #endif
    {   for (auto a : _attribs)
            if (a.location > -1)
                glDisableVertexAttribArray(a.location);
    }

    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
/*! Draws the vertex attributes as a specified primitive type as the vertices
are defined in the attribute arrays.
*/
void SLGLVertexArray::drawArrayAs(SLPrimitive primitiveType,
                                  SLint firstVertex,
                                  SLsizei countVertices)
{   assert(_idVBOAttribs);
    
    #ifndef SL_GLES2
    if (_glHasVAO)
        glBindVertexArray(_idVAO);
    else
    #endif
    {   glBindBuffer(GL_ARRAY_BUFFER, _idVBOAttribs);
        for (auto a : _attribs)
        {   if (a.location > -1)
            {
                // Sets the vertex attribute data pointer to its corresponding GLSL variable
                glVertexAttribPointer(a.location, 
                                      a.elementSize,
                                      GL_FLOAT, 
                                      GL_FALSE, 
                                      _strideBytes, 
                                      (void*)a.offsetBytes);

                // Tell the attribute to be an array attribute instead of a state variable
                glEnableVertexAttribArray(a.location);
            }
        }
    }

    if (countVertices == 0)
        countVertices = _numVertices;

    ////////////////////////////////////////////////////////
    glDrawArrays(primitiveType, firstVertex, countVertices);
    ////////////////////////////////////////////////////////
    
    totalDrawCalls++;
    
    #ifndef SL_GLES2
    if (_glHasVAO)
        glBindVertexArray(0);
    else
    #endif
    {   for (auto a : _attribs)
            if (a.location > -1)
                glDisableVertexAttribArray(a.location);
    }

    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
