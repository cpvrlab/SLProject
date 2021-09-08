//#############################################################################
//  File:      PBR_SkyboxHDR.vert
//  Purpose:   GLSL vertex program for HDR skybox with a cube map
//  Date:      April 2018
//  Authors:   Carlos Arauz, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;  // Vertex position attribute

uniform                  mat4  u_mvpMatrix; // Model-View-Projection matrix

out                      vec3  v_uv1;       // texture coordinate at vertex

//-----------------------------------------------------------------------------
void main()
{
    v_uv1 = normalize(vec3(a_position));
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
