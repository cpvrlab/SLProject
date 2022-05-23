//############################################################################
//  File:      CTVoxelization.vert
//  Purpose:   GLSL vertex shader calculating world-space point used in the
//             voxelization process.
//  Date:      September 2018
//  Authors:   Stefan Thoeni
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//############################################################################

#version 430 core

precision highp float;

//-----------------------------------------------------------------------------
layout(location = 0) in vec4  a_position;    // Vertex position attribute
layout(location = 1) in vec3  a_normal;      // Vertex normal attribute

uniform mat4  u_mMatrix;    // Model matrix (object to world transform)
uniform mat4  u_vMatrix;    // View matrix (world to camera transform)
uniform mat4  u_pMatrix;    // Projection matrix (camera to normalize device coords.)
uniform mat4  u_wsToVs;

out     vec3  v_N_WS;        // Normal at P_VS in world space
out     vec3  v_P_WS;        // position of vertex in world space

out     vec3  v_P_VS;        // position of vertex in world space
//-----------------------------------------------------------------------------
void main(void)
{
  // Careful! pMatrix should be a orthagonal projection!
  v_P_WS = vec3(u_mMatrix * a_position);
  v_N_WS = normalize(mat3(transpose(inverse(u_mMatrix))) * a_normal);
  v_P_VS = vec3(u_wsToVs * u_mMatrix * a_position);

  gl_Position = u_pMatrix * u_vMatrix * u_mMatrix * a_position;
}
//-----------------------------------------------------------------------------
