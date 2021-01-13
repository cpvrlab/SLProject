//#############################################################################
//  File:      PBR_CubeMap.vert
//  Purpose:   GLSL vertex program for rendering cube maps
//  Author:    Carlos Arauz
//  Date:      April 2018
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;     // Vertex position attribute

uniform   mat4  u_mvpMatrix;    // = modelView

out       vec3  v_P_VS;         // sample direction

//-----------------------------------------------------------------------------
void main ()
{
    v_P_VS = a_position.xyz;
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
