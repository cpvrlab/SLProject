//#############################################################################
//  File:      SLGLEnums.h
//  Purpose:   Enumerations containing OpenGL constants
//  Date:      January 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLENUM_H
#define SLGLENUM_H

#include <SL.h>
#include <SLGLState.h>

//-----------------------------------------------------------------------------
//! Enumeration for buffer data types
enum SLGLBufferType
{
    BT_float  = GL_FLOAT,          //!< float vertex attributes
    BT_ubyte  = GL_UNSIGNED_BYTE,  //!< vertex index type (0-2^8)
    BT_ushort = GL_UNSIGNED_SHORT, //!< vertex index type (0-2^16)
    BT_uint   = GL_UNSIGNED_INT    //!< vertex index type (0-2^32)
};
//-----------------------------------------------------------------------------
// Enumeration for OpenGL primitive types
enum SLGLPrimitiveType
{
    PT_points        = GL_POINTS,
    PT_lines         = GL_LINES,
    PT_lineLoop      = GL_LINE_LOOP,
    PT_lineStrip     = GL_LINE_STRIP,
    PT_triangles     = GL_TRIANGLES,
    PT_triangleStrip = GL_TRIANGLE_STRIP,
    PT_triangleFan   = GL_TRIANGLE_FAN
};
//-----------------------------------------------------------------------------
//! Enumeration for float vertex attribute types
/* This index must correspond to the layout location index in GLSL shaders.
 * See also SLMesh vertex attributes.
 */
enum SLGLAttributeType
{
    /*
    // The enum must correspond to the attributes in SLMesh:
    SLVVec3f  P;    //!< Vector for vertex positions                   layout (location = 0)
    SLVVec3f  N;    //!< Vector for vertex normals (opt.)              layout (location = 1)
    SLVVec2f  UV1;  //!< Vector for 1st vertex tex. coords. (opt.)     layout (location = 2)
    SLVVec2f  UV2;  //!< Vector for 2nd. vertex tex. coords. (opt.)    layout (location = 3)
    SLVCol4f  C;    //!< Vector for vertex colors (opt.)               layout (location = 4)
    SLVVec4f  T;    //!< Vector for vertex tangents (opt.)             layout (location = 5)
    SLVVuchar Ji;   //!< 2D Vector of per vertex joint ids (opt.)      layout (location = 6)
    SLVVfloat Jw;   //!< 2D Vector of per vertex joint weights (opt.)  layout (location = 7)
    */

    AT_position = 0,        //!< Vertex position as a 2, 3 or 4 component vectors
    AT_normal,              //!< Vertex normal as a 3 component vector
    AT_uv1,                 //!< Vertex 1st texture coordinate as 2 component vector
    AT_uv2,                 //!< Vertex 2nd texture coordinate as 2 component vector
    AT_color,               //!< Vertex color as 3 or 4 component vector
    AT_tangent,             //!< Vertex tangent as a 4 component vector (see SLMesh)
    AT_jointIndex,          //!< Vertex joint id for vertex skinning
    AT_jointWeight,         //!< Vertex joint weight for vertex skinning

    AT_custom0,             //!< Custom vertex attribute 0
    AT_custom1,             //!< Custom vertex attribute 1
    AT_custom2,             //!< Custom vertex attribute 2
    AT_custom3,             //!< Custom vertex attribute 3
    AT_custom4,             //!< Custom vertex attribute 4
    AT_custom5,             //!< Custom vertex attribute 5
    AT_custom6,             //!< Custom vertex attribute 6
    AT_custom7,             //!< Custom vertex attribute 7
    AT_custom8,             //!< Custom vertex attribute 8
    AT_custom9,             //!< Custom vertex attribute 9

    AT_velocity        = 1, //!< Vertex velocity 3 component vectors
    AT_startTime       = 2, //!< Vertex start time float
    AT_initialVelocity = 3, //!< Vertex initial velocity 3 component vectors
    AT_rotation        = 4, //!< Vertex rotation float
    AT_angularVelo     = 5, //!< Vertex angulare velocity for rotation float
    AT_texNum          = 6, //!< Vertex texture number int
    AT_initialPosition = 7  //!< Vertex initial position 3 component vectors
};
//-----------------------------------------------------------------------------
//! Enumeration for buffer usage types also supported by OpenGL ES
enum SLGLBufferUsage
{
    BU_static  = GL_STATIC_DRAW,  //!< Buffer will be modified once and used many times.
    BU_stream  = GL_STREAM_DRAW,  //!< Buffer will be modified once and used at most a few times.
    BU_dynamic = GL_DYNAMIC_DRAW, //!< Buffer will be modified repeatedly and used many times.
};
//-----------------------------------------------------------------------------

#endif
