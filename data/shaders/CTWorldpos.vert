//############################################################################
//  File:      CTWorldpos.frag
//  Purpose:   GLSL vertex shader calculating world-space point used in the
//             voxelization process.
//  Author:    Stefan Thoeni
//  Date:      September 2018
//  Copyright: Stefan Thoeni
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//############################################################################

#version 430 core
//-----------------------------------------------------------------------------
layout(location = 0) in vec4 a_position;

uniform     mat4  u_mvpMatrix;   // = projection * modelView
uniform     mat4  u_mMatrix;     // model matrix

out vec3 a_P_WS;
//-----------------------------------------------------------------------------
void main(){
	a_P_WS = vec3(u_mMatrix * a_position);
	gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
