//#############################################################################
//  File:      ColorUniform.vert
//  Purpose:   GLSL vertex program for simple uniform color
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4 a_position;  // Vertex position attribute

uniform vec4  u_matDiff;    // uniform color
uniform mat4  u_mMatrix;    // Model matrix (object to world transform)
uniform mat4  u_vMatrix;    // View matrix (world to camera transform)
uniform mat4  u_pMatrix;    // Projection matrix (camera to normalize device coords.)

out     vec4  v_color;      // Resulting color per vertex
//-----------------------------------------------------------------------------
void main(void)
{
    v_color = u_matDiff;
    gl_Position = u_pMatrix * u_vMatrix * u_mMatrix * a_position;
}
//-----------------------------------------------------------------------------
