//#############################################################################
//  File:      ColorUniform.vert
//  Purpose:   GLSL vertex program for simple uniform color
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4 a_position;  // Vertex position attribute

uniform vec4     u_matDiff;      // uniform color
uniform mat4     u_mvpMatrix;       // = projection * modelView

out     vec4     v_color;           // Resulting color per vertex
//-----------------------------------------------------------------------------
void main(void)
{
    v_color = u_matDiff;                   // pass color for interpolation

    gl_Position = u_mvpMatrix * a_position;   // transform vertex position
}
//-----------------------------------------------------------------------------
