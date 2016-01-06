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
#include <SLGLVertexArray.h>

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
      
        //! Updates a buffer object by copying new data or sub-data to the buffer
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
