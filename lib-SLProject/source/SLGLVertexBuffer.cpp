//#############################################################################
//  File:      SLGLVertexBuffer.cpp
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

#include <SLGLVertexBuffer.h>
#include <SLGLProgram.h>
#include <SLScene.h>
#include <half.hpp>

//-----------------------------------------------------------------------------
SLuint SLGLVertexBuffer::totalBufferSize  = 0;
SLuint SLGLVertexBuffer::totalBufferCount = 0;
//-----------------------------------------------------------------------------
//! Constructor initializing with default values
SLGLVertexBuffer::SLGLVertexBuffer()
{   
    _id = 0;
    _numVertices = 0;
    _sizeBytes = 0;
    _outputInterleaved = false;
    _usage = BU_stream;
    _dataType = BT_float;
}
//-----------------------------------------------------------------------------
/*! Deletes the OpenGL objects for the vertex array and the vertex buffer.
The vector _attribs with the attribute information is not cleared.
*/
void SLGLVertexBuffer::deleteGL()
{  
    if (_id)
    {   glDeleteBuffers(1, &_id);
        _id = 0;
        totalBufferCount--;
        totalBufferSize -= _sizeBytes;
    }
}
//-----------------------------------------------------------------------------
void SLGLVertexBuffer::clear(SLGLBufferType dataType) 
{   
   deleteGL(); 
    _attribs.clear(); 
    _dataType = dataType;
}
//-----------------------------------------------------------------------------
SLint SLGLVertexBuffer::attribIndex(SLGLAttributeType type)
{    
    for (SLint i=0; i<_attribs.size(); ++i)
        if (_attribs[i].type == type)
            return i;
    return -1;
}
//-----------------------------------------------------------------------------
/*! Updates the specified vertex attribute. This works only for sequential 
attributes and not for interleaved attributes. This is used e.g. for meshes
with vertex skinning. See SLMesh::draw where we have joint attributes.
*/
void SLGLVertexBuffer::updateAttrib(SLGLAttributeType type, 
                                    SLint elementSize,
                                    void* dataPointer)
{   
    assert(dataPointer && "No data pointer passed");
    assert(elementSize > 0 && elementSize < 5 && "Element size invalid");
    
    // Get attribute index and check element size
    SLint index = attribIndex(type);
    if (index == -1)
        SL_EXIT_MSG("Attribute type does not exist in VBO.");
    if (_attribs[index].elementSize != elementSize)
        SL_EXIT_MSG("Attribute element size differs.");
    if (_outputInterleaved)
        SL_EXIT_MSG("Interleaved buffers can't be updated.");

    // Generate the vertex buffer object if there is none
    if (index && !_id)
        glGenBuffers(1, &_id);

    
    _attribs[index].dataPointer = dataPointer;

    /////////////////////////
    // Convert to Half Floats
    /////////////////////////

    SLVhalf halfs;

    if (_dataType == BT_half)
    {   
        // Create a new array on the heap that must be deleted after glBufferData
        SLint numHalfs = _numVertices * _attribs[index].elementSize;
        halfs.resize(numHalfs);

        // Convert all float to half floats
        for (SLint h=0; h<numHalfs; ++h)
            halfs[h] = half_cast<half>(((SLfloat*)_attribs[index].dataPointer)[h]);

        // Replace the data pointer
        _attribs[index].dataPointer = &halfs[0];
    }
    

    ////////////////////////////////////////////
    // copy sub-data into existing buffer object
    ////////////////////////////////////////////

    glBindBuffer(GL_ARRAY_BUFFER, _id);
    glBufferSubData(GL_ARRAY_BUFFER,
                    _attribs[index].offsetBytes,
                    _attribs[index].bufferSizeBytes,
                    _attribs[index].dataPointer);
    

    ///////////////////////////////////
    // Delete the converted half floats
    ///////////////////////////////////

    if (_dataType == BT_half)
    {   halfs.clear();
        _attribs[index].dataPointer = 0;
    }

    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
/*! Generates the OpenGL VBO for one or more vertex attributes. 
If the input data is an interleaved array (all attribute data pointer where 
identical) also the output buffer will be generated as an interleaved array. 
Vertex arrays with attributes that are updated can not be interleaved. 
Vertex attributes with separate arrays can generate an interleaved or a 
sequential vertex buffer.\n\n
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
void SLGLVertexBuffer::generate(SLuint numVertices, 
                                SLGLBufferUsage usage,
                                SLbool outputInterleaved)
{   
    assert(numVertices);

    // if buffers exist delete them first
    deleteGL();

    _numVertices = numVertices;
    _usage = usage;
    _outputInterleaved = outputInterleaved;
    
    // Generate the vertex buffer object
    if (_attribs.size())
    {   glGenBuffers(1, &_id);
        glBindBuffer(GL_ARRAY_BUFFER, _id);
    }

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

    _sizeBytes = 0;
    _strideBytes = 0;

    if (inputIsInterleaved)
    {   _outputInterleaved = true;
        for (SLint i=0; i<_attribs.size(); ++i) 
        {   SLuint elementSizeBytes = _attribs[i].elementSize * sizeOfType(_dataType);
            _attribs[i].offsetBytes = _strideBytes;
            _attribs[i].bufferSizeBytes = elementSizeBytes * _numVertices;
            _sizeBytes += _attribs[i].bufferSizeBytes;
            _strideBytes += elementSizeBytes;
        }  
    }
    else // input is in separate attribute data blocks
    {
        for (SLint i=0; i<_attribs.size(); ++i) 
        {   SLuint elementSizeBytes = _attribs[i].elementSize * sizeOfType(_dataType);
            if (_outputInterleaved)
                 _attribs[i].offsetBytes = _strideBytes;
            else _attribs[i].offsetBytes = _sizeBytes;
            _attribs[i].bufferSizeBytes = elementSizeBytes * _numVertices;
            _sizeBytes += _attribs[i].bufferSizeBytes;
            if (_outputInterleaved) _strideBytes += elementSizeBytes;
        }
    }


    /////////////////////////
    // Convert to Half Floats
    /////////////////////////

    if (_dataType == BT_half)
    {   for (SLint i=0; i < _attribs.size(); ++i)
        {   // Create a new array on the heap that must be deleted after glBufferData
            SLint numHalfs = _numVertices * _attribs[i].elementSize;
            SLhalf* pHalfs = new SLhalf[numHalfs];

            // Convert all float to half floats
            for (SLint h=0; h<numHalfs; ++h)
                pHalfs[h] = half_cast<half>(((SLfloat*)_attribs[i].dataPointer)[h]);

            // Replace the data pointer
            _attribs[i].dataPointer = pHalfs;
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
                                      _dataType,
                                      GL_FALSE, 
                                      _strideBytes,
                                      (void*)(size_t)a.offsetBytes);
        
                // Tell the attribute to be an array attribute instead of a state variable
                glEnableVertexAttribArray(a.location);
            }
        }

        // generate the interleaved VBO buffer on the GPU
        glBufferData(GL_ARRAY_BUFFER, _sizeBytes, _attribs[0].dataPointer, _usage);
    }
    else  // input is in separate attribute data block
    {    
        if (_outputInterleaved) // Copy attribute data interleaved
        {
            SLVuchar data; 
            data.resize(_sizeBytes);

            for (auto a : _attribs)
            {   
                SLuint elementSizeBytes = a.elementSize * sizeOfType(_dataType);

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
                                          _dataType,
                                          GL_FALSE, 
                                          _strideBytes,
                                          (void*)(size_t)a.offsetBytes);
        
                    // Tell the attribute to be an array attribute instead of a state variable
                    glEnableVertexAttribArray(a.location);

                }
            }

            // generate the interleaved VBO buffer on the GPU
            glBufferData(GL_ARRAY_BUFFER, _sizeBytes, &data[0], _usage);
        } 
        else // copy attributes buffers sequentially
        {   
            // allocate the VBO buffer on the GPU
            glBufferData(GL_ARRAY_BUFFER, _sizeBytes, NULL, _usage);
        
            for (auto a : _attribs)
            {   if (a.location > -1)
                {
                    // Copies the attributes data at the right offset into the VBO
                    glBufferSubData(GL_ARRAY_BUFFER,
                                    a.offsetBytes,
                                    a.bufferSizeBytes,
                                    a.dataPointer);
        
                    // Sets the vertex attribute data pointer to its corresponding GLSL variable
                    glVertexAttribPointer(a.location, 
                                          a.elementSize, 
                                          _dataType,
                                          GL_FALSE, 
                                          0,
                                          (void*)(size_t)a.offsetBytes);
        
                    // Tell the attribute to be an array attribute instead of a state variable
                    glEnableVertexAttribArray(a.location);
                }
            }
        }
    }

    totalBufferCount++;
    totalBufferSize += _sizeBytes;


    ///////////////////////////////////
    // Delete the converted half floats
    ///////////////////////////////////

    if (_dataType == BT_half)
    {   for (SLint i=0; i < _attribs.size(); ++i)
        {   delete[] _attribs[i].dataPointer;
            _attribs[i].dataPointer = 0;
        }
    }
    
    #ifdef _GLDEBUG
    GET_GL_ERROR;
    #endif
}
//-----------------------------------------------------------------------------
/*! This method is only used by SLGLVertexArray drawing methods for OpenGL
contexts prior to 3.0 where vertex array objects did not exist. This is the 
additional overhead that had to be done per draw call.
*/
void SLGLVertexBuffer::bindAndEnableAttrib()
{
    if (_attribs.size())
    {
        glBindBuffer(GL_ARRAY_BUFFER, _id);

        for (auto a : _attribs)
        {   if (a.location > -1)
            {
                // Sets the vertex attribute data pointer to its corresponding GLSL variable
                glVertexAttribPointer(a.location, 
                                        a.elementSize,
                                        _dataType, 
                                        GL_FALSE, 
                                        _strideBytes, 
                                        (void*)(size_t)a.offsetBytes);

                // Tell the attribute to be an array attribute instead of a state variable
                glEnableVertexAttribArray(a.location);
            }
        }
    }
}
//-----------------------------------------------------------------------------
/*! This method is only used by SLGLVertexArray drawing methods for OpenGL
contexts prior to 3.0 where vertex array objects did not exist. This is the 
additional overhead that had to be done per draw call.
*/
void SLGLVertexBuffer::disableAttrib()
{
    if (_attribs.size())
    {   for (auto a : _attribs)
            if (a.location > -1)
                glDisableVertexAttribArray(a.location);
    }
}
//-----------------------------------------------------------------------------
SLint SLGLVertexBuffer::sizeOfType(SLGLBufferType type)
{
    switch (type)
    {   case BT_half :  return sizeof(half);
        case BT_float:  return sizeof(float);
        case BT_ubyte:  return sizeof(unsigned char);
        case BT_ushort: return sizeof(unsigned short);
        case BT_uint:   return sizeof(unsigned int);
        default: SL_EXIT_MSG("Invalid buffer data type");
    }
    return 0;
}
//-----------------------------------------------------------------------------
