//#############################################################################
//  File:      FontTex.vert
//  Purpose:   GLSL vertex program for textured font
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4 a_position; // Vertex position attribute
layout (location = 2) in vec3 a_texCoord; // Vertex texture coord. attribute

uniform mat4  u_mMatrix;    // Model matrix (object to world transform)
uniform mat4  u_vMatrix;    // View matrix (world to camera transform)
uniform mat4  u_pMatrix;    // Projection matrix (camera to normalize device coords.)

out     vec2  v_texCoord;   // texture coordinate at vertex
//-----------------------------------------------------------------------------
void main()
{
    v_texCoord = a_texCoord.xy;
    gl_Position = u_pMatrix * u_vMatrix * u_mMatrix * a_position;
}
//-----------------------------------------------------------------------------
