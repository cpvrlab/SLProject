//############################################################################
//  File:      CT-Tex.vert
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
layout(location = 2) in vec2  a_texCoord;    // Vertex texture coordinate attribute

uniform mat4  u_mvpMatrix;   // = projection * modelView
uniform mat4  u_mMatrix;     // Model matrix (object to world transform)
uniform mat4  u_wsToVs;      // convert from ws to voxel space

out		vec3  o_N_WS;        // Normal at P_VS in world space
out		vec3  o_P_VS;        // position of vertex in world space
out		vec3  o_P_WS;        // position of vertex in world space
out     vec2  o_Tc;          // Texture coordinate output
//-----------------------------------------------------------------------------
void main(void)
{
	o_P_WS = vec3(u_mMatrix * a_position);
	o_P_VS = vec3(u_wsToVs * u_mMatrix * a_position);
	o_N_WS = normalize(mat3(transpose(inverse(u_mMatrix))) * a_normal);
    o_Tc   = a_texCoord;

	gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
