//#############################################################################
//  File:      SLGLBuffer.h
//  Purpose:   Wrapper class around OpenGL Vertex Buffer Objects 
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLBUFFER_H
#define SLGLBUFFER_H

#include <stdafx.h>

//-----------------------------------------------------------------------------
//! Enumeration for buffer data types
typedef enum
{
    SL_FLOAT          = GL_FLOAT,          // vertex attributes (position, normals)
    SL_UNSIGNED_BYTE  = GL_UNSIGNED_BYTE,  // vertex index type (0-2^8)
    SL_UNSIGNED_SHORT = GL_UNSIGNED_SHORT, // vertex index type (0-2^16)
    SL_UNSIGNED_INT   = GL_UNSIGNED_INT    // vertex index type (0-2^32)
} SLBufferType;
//-----------------------------------------------------------------------------
//! Enumeration for buffer target types
typedef enum
{
    SL_ARRAY_BUFFER         = GL_ARRAY_BUFFER,         // vertex attributes arrays
    SL_ELEMENT_ARRAY_BUFFER = GL_ELEMENT_ARRAY_BUFFER  // vertex index arrays
} SLBufferTarget;
//-----------------------------------------------------------------------------
/*! Enumeration for buffer usage types also supported by OpenGL ES
STATIC:  Buffer contents will be modified once and used many times.
STREAM:  Buffer contents will be modified once and used at most a few times.
DYNAMIC: Buffer contents will be modified repeatedly and used many times.
*/
typedef enum
{
    SL_STATIC_DRAW  = GL_STATIC_DRAW,
    SL_STREAM_DRAW  = GL_STREAM_DRAW,
    SL_DYNAMIC_DRAW = GL_DYNAMIC_DRAW,
} SLBufferUsage;
//-----------------------------------------------------------------------------
// Enumeration for OpenGL primitive types also available on OpenGL ES
typedef enum
{
    SL_POINTS         = GL_POINTS,
    SL_LINES          = GL_LINES,
    SL_LINE_LOOP      = GL_LINE_LOOP,
    SL_LINE_STRIP     = GL_LINE_STRIP,
    SL_TRIANGLES      = GL_TRIANGLES,
    SL_TRIANGLE_STRIP = GL_TRIANGLE_STRIP,
    SL_TRIANGLE_FAN   = GL_TRIANGLE_FAN
} SLPrimitive;
//-----------------------------------------------------------------------------
//! Encapsulation of an OpenGL buffer object
/*! 
The SLGLBuffer class encapsulate all functionality for geometry drawing with 
core profile OpenGL. ALL drawing is done with vertex buffer objects (VBO).
There is no more Immediate Mode rendering from the old OpenGL (compatibility
profile) and no more client memory access by the OpenGL subsystem.
*/
class SLGLBuffer
{
    public:
      
        //! Default constructor
        SLGLBuffer();
      
        //! Destructor calling dispose
        ~SLGLBuffer();
      
        //! Deletes the buffer object
        void dispose();
      
        //! Getter of the OpenGL buffer id
        SLint id() {return _id;}
                              
        //! Generic buffer generation (see private members for details)
        void generate(void* dataPointer, 
                     SLuint numElements, 
                     SLint elementSize, 
                     SLBufferType   type   = SL_FLOAT,        
                     SLBufferTarget target = SL_ARRAY_BUFFER,
                     SLBufferUsage  usage  = SL_STATIC_DRAW);
      
        //! Updates a buffer object by copying new data or subdata to the buffer
        void update(const void* dataPointer, 
                    SLuint numElements, 
                    SLuint offsetElements = 0);
      
        //! Binds the buffer and enables the GLSL attribute by index 
        void bindAndEnableAttrib(SLint attribIndex,
                                 SLuint dataOffsetBytes = 0,
                                 SLint stride = 0);
                                     
        //! Binds the buffer and draws the elements with a primitive type
        void bindAndDrawElementsAs(SLPrimitive primitiveType,
                                   SLuint numIndexes = 0,
                                   SLuint indexOffsetBytes = 0);
      
        //! Draws a vertex array directly with a primitive
        void drawArrayAs(SLPrimitive primitiveType,
                         SLuint indexFirstVertex = 0,
                         SLuint numVertices = 0);

        //!  Draws a vertex array with constant color attribute
        void drawArrayAsConstantColor(SLPrimitive primitiveType,
                                     SLCol4f color,
                                     SLfloat lineOrPointWidth = 1.0f,
                                     SLuint  indexFirstVertex = 0,
                                     SLuint  numVertices = 0);

        //! Draws a vertex array as lines with constant color attribute
        void drawArrayAsConstantColorLines(SLCol3f color,
                                           SLfloat lineSize = 1.0f,
                                           SLuint  indexFirstVertex = 0,
                                           SLuint  numVertices = 0);

        //! Draws a vertex array as line strip with constant color attribute
        void drawArrayAsConstantColorLineStrip(SLCol3f color,
                                               SLfloat lineSize = 1.0f,
                                               SLuint  indexFirstVertex = 0,
                                               SLuint  numVertices = 0);
                                         
        //! Draws a vertex array as points with constant color attribute
        void drawArrayAsConstantColorPoints(SLCol4f color,
                                            SLfloat pointSize = 1.0f,
                                            SLuint  indexFirstVertex = 0,
                                            SLuint  numVertices = 0);
        //! disables vertex attribute array
        void disableAttribArray();

        // Some statistics
        static SLuint totalBufferCount;  //! static total no. of buffers in use
        static SLuint totalBufferSize;   //! static total size of all buffers in bytes
        static SLuint totalDrawCalls;    //! static total no. of draw calls

        // Getters
        SLuint         numElements() {return _numElements;}
        SLint          elementSize() {return _elementSize;}
        SLint          typeSize()    {return _typeSize;}
                                               
    private:               
        SLuint         _id;           //! OpenGL id of the buffer object
        SLuint         _numElements;  //! No. of elements in the array
        SLint          _elementSize;  //! Size of one element in the array
        SLint          _typeSize;     //! Size of raw data type in bytes
        SLBufferType   _dataTypeGL;   //! OpenGL data type (default FLOAT)
        SLBufferTarget _targetTypeGL; //! target type (default ARRAY_BUFFER)
        SLBufferUsage  _usageTypeGL;  //! usage type (default STATIC_DRAW)
        SLint          _attribIndex;  //! vertex attribute index in active shader
};
//-----------------------------------------------------------------------------

#endif
