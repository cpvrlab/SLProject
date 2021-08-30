//#############################################################################
//  File:      OculusScene.vert
//  Purpose:   Oculus Rift Distortion Shader
//  Date:      November 2013
//  Authors:   Marc Wacker, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;  // Vertex position attribute
layout (location = 1) in vec3  a_normal;    // Vertex normal attribute
layout (location = 2) in vec2  a_uv1; 		// Vertex texture attribute
layout (location = 4) in vec4  a_color;     // Vertex color attribute

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_camProjection;

uniform mat4 u_mv;
uniform mat4 u_mvp;

out     vec3 v_normal;
out     vec2 v_uv1;
out     vec4 v_color;
out     vec3 v_lightDir;	//light direction in WORLD SPACE!
//-----------------------------------------------------------------------------
void main()
{	
	gl_Position = u_mvp * vec4(a_position, 1.0);
	v_normal = normalize((u_model * vec4(a_normal, 0)).xyz);
	v_uv1 = a_uv1;
	v_color = a_color;
	v_lightDir = vec3(1,-1,-1);
}
//-----------------------------------------------------------------------------