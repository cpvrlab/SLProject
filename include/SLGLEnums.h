//#############################################################################
//  File:      SLGLEnums.h
//  Purpose:   Enumerations containing OpenGL constants 
//  Author:    Marcus Hudritsch
//  Date:      January 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLENUM_H
#define SLGLENUM_H

#include <SL.h>

//-----------------------------------------------------------------------------
//! Enumeration for buffer data types
enum SLGLBufferType
{
    BT_float  = GL_FLOAT,               //!< float vertex attributes
    BT_ubyte  = GL_UNSIGNED_BYTE,       //!< vertex index type (0-2^8)
    BT_ushort = GL_UNSIGNED_SHORT,      //!< vertex index type (0-2^16)
    BT_uint   = GL_UNSIGNED_INT         //!< vertex index type (0-2^32)
};
//-----------------------------------------------------------------------------
// Enumeration for OpenGL primitive types
enum SLGLPrimitiveType
{   PT_points        = GL_POINTS,
    PT_lines         = GL_LINES,
    PT_lineLoop      = GL_LINE_LOOP,
    PT_lineStrip     = GL_LINE_STRIP,
    PT_triangles     = GL_TRIANGLES,
    PT_triangleStrip = GL_TRIANGLE_STRIP,
    PT_triangleFan   = GL_TRIANGLE_FAN
};
//-----------------------------------------------------------------------------
//! Enumeration for float vertex attribute types
enum SLGLAttributeType
{   AT_position,    //!< Vertex position as a 2, 3 or 4 component vectors
    AT_normal,      //!< Vertex normal as a 3 component vector
    AT_texCoord,    //!< Vertex texture coordinate as 2 component vector
    AT_tangent,     //!< Vertex tangent as a 4 component vector (see SLMesh) 
    AT_jointWeight, //!< Vertex joint weight for vertex skinning
    AT_jointIndex,  //!< Vertex joint id for vertex skinning
    AT_color,       //!< Vertex color as 3 or 4 component vector
    AT_custom0,     //!< Custom vertex attribute 0
    AT_custom1,     //!< Custom vertex attribute 1
    AT_custom2,     //!< Custom vertex attribute 2
    AT_custom3,     //!< Custom vertex attribute 3
    AT_custom4,     //!< Custom vertex attribute 4
    AT_custom5,     //!< Custom vertex attribute 5
    AT_custom6,     //!< Custom vertex attribute 6
    AT_custom7,     //!< Custom vertex attribute 7
    AT_custom8,     //!< Custom vertex attribute 8
    AT_custom9      //!< Custom vertex attribute 0
};
//-----------------------------------------------------------------------------
//! Enumeration for buffer usage types also supported by OpenGL ES
enum SLGLBufferUsage
{   BU_static  = GL_STATIC_DRAW,        //!< Buffer will be modified once and used many times.
    BU_stream  = GL_STREAM_DRAW,        //!< Buffer will be modified once and used at most a few times.
    BU_dynamic = GL_DYNAMIC_DRAW,       //!< Buffer will be modified repeatedly and used many times.
};
//-----------------------------------------------------------------------------

#endif
