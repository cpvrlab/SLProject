//#############################################################################
//  File:      SkyBox.vert
//  Purpose:   GLSL vertex program for unlit skybox with a cube map
//  Date:      October 2017
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;     // Vertex position attribute

uniform mat4  u_mMatrix;    // Model matrix (object to world transform)
uniform mat4  u_vMatrix;    // View matrix (world to camera transform)
uniform mat4  u_pMatrix;    // Projection matrix (camera to normalize device coords.)

out     vec3  v_uv0;        // texture coordinate at vertex
//-----------------------------------------------------------------------------
void main()
{
    v_uv0 = normalize(vec3(a_position));
   
    // Set the transformes vertex position   
    gl_Position = u_pMatrix * u_vMatrix * u_mMatrix * a_position;
}
//-----------------------------------------------------------------------------
