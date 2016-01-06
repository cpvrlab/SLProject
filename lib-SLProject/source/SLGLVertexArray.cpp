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
}
//-----------------------------------------------------------------------------
/*! Destructor calling dispose
*/
SLGLVertexArray::~SLGLVertexArray() 
{
    dispose();
}
//-----------------------------------------------------------------------------
/*! Deletes all data
*/
void SLGLVertexArray::dispose()
{  
    if (_glHasVAO && _idVAO) 
    {   glDeleteVertexArrays(1, &_idVAO);
        _idVAO = 0;
    }
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
    
    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
void SLGLVertexArray::addAttrib(SLVertexAttribType aType, 
                                SLint elementSize,
                                SLint location, 
                                void* dataPointer)
{   assert(dataPointer);
    assert(elementSize);

    if (aType == SL_POSITION && location == -1)
        SL_EXIT_MSG("The position attribute has no variable location.");

    if (attribIndex(aType) >= 0)
        SL_EXIT_MSG("Attribute type already exists.");

    SLVertexAttrib va;
    va.type = aType;
    va.elementSize = elementSize;
    va.dataPointer = dataPointer;
    va.location = location;

    _attribs.push_back(va);
}
//-----------------------------------------------------------------------------
void SLGLVertexArray::addIndices(SLuint numIndices,
                                 SLBufferType indexDataType,
                                 void* dataPointer)
{   assert(numIndices);
    assert(dataPointer);
    assert(indexDataType);
    
    if (indexDataType == SL_UNSIGNED_SHORT && numIndices > 65535)
        SL_EXIT_MSG("Index data type not sufficient.");
    if (indexDataType == SL_UNSIGNED_BYTE && numIndices > 255)
        SL_EXIT_MSG("Index data type not sufficient.");
        
    _numIndices = numIndices;
    _indexDataType = indexDataType;
    _indexData = dataPointer;

    switch (indexDataType)
    {   case GL_UNSIGNED_BYTE:  _indexTypeSize = sizeof(GLubyte);  break;
        case GL_UNSIGNED_SHORT: _indexTypeSize = sizeof(GLushort); break;
        case GL_UNSIGNED_INT:   _indexTypeSize = sizeof(GLuint);   break;
        default: SL_EXIT_MSG("Invalid index data type");
    }
}
//-----------------------------------------------------------------------------
void SLGLVertexArray::updateAttrib(SLVertexAttribType aType, 
                                   SLint elementSize,
                                   void* dataPointer)
{   assert(dataPointer);
    assert(elementSize > 1 && elementSize < 5);
    
    // Get attribute index and check element size
    SLint index = attribIndex(aType);
    if (index == -1)
        SL_EXIT_MSG("Attribute type does not exist in VAO.");
    if (_attribs[index].elementSize != elementSize)
        SL_EXIT_MSG("Attribute element size differs.");
    
    if (_glHasVAO)
        glBindVertexArray(_idVAO);
    
    // copy sub-data into existing buffer object
    glBindBuffer(GL_ARRAY_BUFFER, _idVBOAttribs);
    glBufferSubData(GL_ARRAY_BUFFER,
                    _attribs[index].bufferOffsetBytes,
                    _attribs[index].bufferSizeBytes,
                    dataPointer);
    
    if (_glHasVAO)
        glBindVertexArray(0);
    
    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
/*!
*/
void SLGLVertexArray::generate(SLuint numVertices, 
                               SLBufferUsage usage)
{   assert(numVertices);

    // if buffers exist delete them first
    dispose();

    _numVertices = numVertices;
    _usage = usage;

    // Generate and bind VAO
    if (_glHasVAO)
    {   glGenVertexArrays(1, &_idVAO);
        glBindVertexArray(_idVAO);
    }
    glGenBuffers(1, &_idVBOAttribs);
    glBindBuffer(GL_ARRAY_BUFFER, _idVBOAttribs);

    // calculate total vbo size and attribute offset
    _vboSize = 0;
    for (SLint i=0; i<_attribs.size(); ++i) 
    {   _attribs[i].bufferOffsetBytes = _vboSize;
        _attribs[i].bufferSizeBytes = _attribs[i].elementSize * sizeof(SLfloat) * _numVertices;
        _vboSize += _attribs[i].bufferSizeBytes;
    }

    // allocate the vbo buffer on the GPU
    glBufferData(GL_ARRAY_BUFFER, _vboSize, NULL, _usage);
    totalBufferCount++;
    totalBufferSize += _vboSize;

    for (auto a : _attribs)
    {   if (a.location > -1)
        {
            // Copies the attributes data at the right offset into the vbo
            glBufferSubData(GL_ARRAY_BUFFER,
                            a.bufferOffsetBytes,
                            a.bufferSizeBytes,
                            a.dataPointer);
        
            // Sets the vertex attribute data pointer to its corresponding GLSL variable
            glVertexAttribPointer(a.location, 
                                  a.elementSize, 
                                  GL_FLOAT,
                                  GL_FALSE, 
                                  0,
                                  (void*)a.bufferOffsetBytes);
        
            // Tell the attribute to be an array attribute instead of a state variable
            glEnableVertexAttribArray(a.location);
        }
    }

    // create vbo for indices and copy its data to the GPU
    if (_numIndices)
    {   glGenBuffers(1, &_idVBOIndices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idVBOIndices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, _numIndices * _indexTypeSize, _indexData, _usage);
        totalBufferCount++;
        totalBufferSize += _numIndices * _indexTypeSize;
    }

    if (_glHasVAO)
        glBindVertexArray(0);

    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}      
//-----------------------------------------------------------------------------
void SLGLVertexArray::drawElementsAs(SLPrimitive primitiveType,
                                     SLuint numIndexes,
                                     SLuint indexOffset)
{   assert(_idVBOAttribs);
    assert(_numIndices && _idVBOIndices);

    if (_glHasVAO)
        glBindVertexArray(_idVAO);
    else
    {   glBindBuffer(GL_ARRAY_BUFFER, _idVBOAttribs);
        for (auto a : _attribs)
        {   if (a.location > -1)
            {
                // Sets the vertex attribute data pointer to its corresponding GLSL variable
                glVertexAttribPointer(a.location, 
                                      a.elementSize,
                                      GL_FLOAT, 
                                      GL_FALSE, 
                                      0, 
                                      (void*)a.bufferOffsetBytes);

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

    if (_glHasVAO)
        glBindVertexArray(0);
    else
    {   for (auto a : _attribs)
            if (a.location > -1)
                glDisableVertexAttribArray(a.location);
    }

    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
void SLGLVertexArray::drawArrayAs(SLPrimitive primitiveType,
                                  SLint firstVertex,
                                  SLsizei countVertices)
{   assert(_idVBOAttribs);

    if (_glHasVAO)
        glBindVertexArray(_idVAO);
    else
    {   glBindBuffer(GL_ARRAY_BUFFER, _idVBOAttribs);
        for (auto a : _attribs)
        {   if (a.location > -1)
            {
                // Sets the vertex attribute data pointer to its corresponding GLSL variable
                glVertexAttribPointer(a.location, 
                                      a.elementSize,
                                      GL_FLOAT, 
                                      GL_FALSE, 
                                      0, 
                                      (void*)a.bufferOffsetBytes);

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

    if (_glHasVAO)
        glBindVertexArray(0);
    else
    {   for (auto a : _attribs)
            if (a.location > -1)
                glDisableVertexAttribArray(a.location);
    }

    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
void SLGLVertexArray::generateLineVertices(SLuint numVertices,
                                           SLint elementSize,
                                           void* dataPointer)
{
    assert(dataPointer);
    assert(elementSize);
    assert(numVertices);
    
    SLGLProgram* sp = SLScene::current->programs(ColorUniform);
    SLGLState* state = SLGLState::getInstance();
    sp->useProgram();
    SLint location = sp->getAttribLocation("a_position");
    
    if (location == -1)
        SL_EXIT_MSG("The position attribute has no variable location.");
    
    // Add attribute if it doesn't exist
    if (attribIndex(SL_POSITION) == -1)
    {   addAttrib(SL_POSITION, elementSize, location, dataPointer);
        generate(numVertices);
    } else
        updateAttrib(SL_POSITION, elementSize, dataPointer);
}
//-----------------------------------------------------------------------------
/*! Draws the vertex position as line primitive with constant color
*/
void SLGLVertexArray::drawColorLines(SLCol3f color,
                                     SLfloat lineWidth,
                                     SLuint  indexFirstVertex,
                                     SLuint  numVertices)
{   assert(_idVBOAttribs);
    assert(numVertices <= _numVertices);
   
    // Prepare shader
    SLMaterial::current = 0;
    SLGLProgram* sp = SLScene::current->programs(ColorUniform);
    SLGLState* state = SLGLState::getInstance();
    sp->useProgram();
    sp->uniformMatrix4fv("u_mvpMatrix", 1, (SLfloat*)state->mvpMatrix());
   
    // Set uniform color
    glUniform4fv(sp->getUniformLocation("u_color"), 1, (SLfloat*)&color);
   
    #ifndef SL_GLES2
    if (lineWidth!=1.0f)
        glLineWidth(lineWidth);
    #endif
                
    drawArrayAs(SL_LINES, indexFirstVertex, numVertices);
   
    #ifndef SL_GLES2
    if (lineWidth!=1.0f)
        glLineWidth(1.0f);
    #endif
    
    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
/*! Draws a vertex array buffer as line primitive with constant color attribute
*/
void SLGLVertexArray::drawColorPoints(SLCol4f color,
                                      SLfloat pointSize,
                                      SLuint  indexFirstVertex,
                                      SLuint  numVertices)
{   assert(_idVBOAttribs);
    assert(numVertices <= _numVertices);
   
    // Prepare shader
    SLMaterial::current = 0;
    SLGLProgram* sp = SLScene::current->programs(ColorUniform);
    SLGLState* state = SLGLState::getInstance();
    sp->useProgram();
    sp->uniformMatrix4fv("u_mvpMatrix", 1, (SLfloat*)state->mvpMatrix());
   
    // Set uniform color           
    SLint indexC = sp->getUniformLocation("u_color");
    glUniform4fv(indexC, 1, (SLfloat*)&color);
   
    glBindBuffer(GL_ARRAY_BUFFER, _idVBOAttribs);

    SLint posLoc = sp->getAttribLocation("a_position");

    if (_glHasVAO)
        glBindVertexArray(_idVAO);

    for (auto a : _attribs)
    {   if (a.type == SL_POSITION)
        {
            // Sets the vertex attribute data pointer to its corresponding GLSL variable
            glVertexAttribPointer(posLoc, 
                                  a.elementSize,
                                  GL_FLOAT, 
                                  GL_FALSE, 
                                  0, 
                                  (void*)a.bufferOffsetBytes);

            // Tell the attribute to be an array attribute instead of a state variable
            glEnableVertexAttribArray(posLoc);
        }
    }

    // Activate the index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idVBOIndices);

    #ifndef SL_GLES2
    if (pointSize!=1.0f)
        glPointSize(pointSize);
    #endif

    ///////////////////////////////////////////////////////
    glDrawArrays(GL_POINTS, 
                 indexFirstVertex, 
                 numVertices ? numVertices : _numVertices);
    ///////////////////////////////////////////////////////

    totalDrawCalls++;

    if (_glHasVAO)
        glBindVertexArray(0);
   
    #ifndef SL_GLES2
    if (pointSize!=1.0f)
        glPointSize(1.0f);
    #endif
    
    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
SLint SLGLVertexArray::attribIndex(SLVertexAttribType aType)
{    
    for (SLint i=0; i<_attribs.size(); ++i)
        if (_attribs[i].type == aType)
            return i;
    return -1;
}
//-----------------------------------------------------------------------------
