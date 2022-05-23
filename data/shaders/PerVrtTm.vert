//#############################################################################
//  File:      PerVrtTm.vert
//  Purpose:   GLSL vertex program for texture mapping only
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;  // Vertex position attribute
layout (location = 2) in vec2  a_uv0;       // Vertex texture attribute

uniform mat4  u_mMatrix;    // Model matrix (object to world transform)
uniform mat4  u_vMatrix;    // View matrix (world to camera transform)
uniform mat4  u_pMatrix;    // Projection matrix (camera to normalize device coords.)

out     vec3    v_P_VS;         // Point of illumination in view space (VS)
out     vec2    v_uv0;          // texture coordinate at vertex
//-----------------------------------------------------------------------------
void main()
{
    // out position in view space
    mat4 mvMatrix = u_vMatrix * u_mMatrix;
    v_P_VS = vec3(mvMatrix * a_position);

    // Set the texture coord. output for interpolated tex. coords.
    v_uv0 = a_uv0.xy;
   
    // Set the transformes vertex position
    gl_Position = u_pMatrix * mvMatrix * a_position;
}
//-----------------------------------------------------------------------------
