//#############################################################################
//  File:      PBR_CubeMap.vert
//  Purpose:   GLSL vertex program for rendering cube maps
//  Date:      April 2018
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;  // Vertex position attribute

uniform                  mat4  u_mvpMatrix; // Model-View-Projection matrix

out                      vec3  v_P_WS;      // texture coordinate at vertex
//-----------------------------------------------------------------------------
void main ()
{
    v_P_WS = a_position.xyz;
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
