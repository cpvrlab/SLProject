//#############################################################################
//  File:      PBR_BRDFIntegration.vert
//  Purpose:   GLSL vertex program for generating a BRDF integration map, which
//             is the second part of the specular integral.
//  Author:    Carlos Arauz
//  Date:      April 2018
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;  // Vertex position attribute

uniform                  mat4  u_mvpMatrix; // Model-View-Projection matrix

out                      vec2  v_P_WS;      // texture coordinate at vertex
//-----------------------------------------------------------------------------
void main()
{
    v_P_WS = a_position.xyz;
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
