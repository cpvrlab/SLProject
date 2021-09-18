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

uniform mat4    u_mvMatrix;     // modelview matrix
uniform mat4    u_mvpMatrix;    // = projection * modelView

out     vec3    v_P_VS;         // Point of illumination in view space (VS)
out     vec2    v_uv0;          // texture coordinate at vertex
//-----------------------------------------------------------------------------
void main()
{
    // out position in view space
    v_P_VS = vec3(u_mvMatrix * a_position);

    // Set the texture coord. output for interpolated tex. coords.
    v_uv0 = a_uv0.xy;
   
    // Set the transformes vertex position
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
