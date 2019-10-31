//#############################################################################
//  File:      Voxelization.geom
//  Purpose:   GLSL geometry shader projects triangle onto main axis and
//             projects to clip space for voxelization
//  Author:    Stefan Thöni
//  Date:      September 2018
//  Copyright: Stefan Thöni
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################
#version 430 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 v_N_WS[];
in vec3 v_P_WS[];
in vec3 v_P_VS[];

out vec3 o_F_WS; // Fragment world position
out vec3 o_F_VS; // Fragment position in voxel space
out vec3 o_N_WS; // Fragment normal

//-----------------------------------------------------------------------------
void main(void)
{
  // calculate face normal:
  const vec3 p1 = v_P_VS[1] - v_P_VS[0];
  const vec3 p2 = v_P_VS[2] - v_P_VS[0];
  const vec3 faceN = abs(cross(p1, p2));

  // Main axis of the triangle:
  uint maxAxis = faceN[1] > faceN[0] ? 1 : 0;
  maxAxis = faceN[2] > faceN[maxAxis] ? 2 : maxAxis;

  // emit voxel position:
  for(uint i = 0; i < 3; ++i){
    o_F_WS = v_P_WS[i];
    o_N_WS = v_N_WS[i];
    o_F_VS = v_P_VS[i];

    if(maxAxis == 0){
      gl_Position = vec4(o_F_VS.z, o_F_VS.y, 0, 1);
    } else if (maxAxis == 1){
      gl_Position = vec4(o_F_VS.x, o_F_VS.z, 0, 1);
    } else {
      gl_Position = vec4(o_F_VS.x, o_F_VS.y, 0, 1);
    }
    EmitVertex();
  }
  EndPrimitive();
}
//-----------------------------------------------------------------------------
