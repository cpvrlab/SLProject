//#############################################################################
//  File:      OculusScene.vert
//  Purpose:   Oculus Rift Distortion Shader
//  Author:    Marc Wacker, Roman Kuehne
//  Date:      November 2013
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

//-----------------------------------------------------------------------------
in      vec3 a_position;
in      vec3 a_normal;
in      vec2 a_texCoord;
in      vec4 a_color;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

uniform mat4 u_mv;
uniform mat4 u_mvp;

out     vec3 v_normal;
out     vec2 v_texCoord;
out     vec4 v_color;
out     vec3 v_lightDir;	//light direction in WORLD SPACE!
//-----------------------------------------------------------------------------
void main()
{	
	gl_Position = u_mvp * vec4(a_position, 1.0);
	v_normal = normalize((u_model * vec4(a_normal, 0)).xyz);
	v_texCoord = a_texCoord;
	v_color = a_color;
	v_lightDir = vec3(1,-1,-1);
}
//-----------------------------------------------------------------------------