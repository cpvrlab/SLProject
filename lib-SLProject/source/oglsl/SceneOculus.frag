//#############################################################################
//  File:      SceneOculus.frag
//  Purpose:   Oculus Rift Distortion Shader
//  Author:    Marc Wacker, Roman Kühne
//  Date:      November 2013
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

varying vec2 v_texCoord;
varying vec4 v_color;
varying vec3 v_normal;
varying vec3 v_lightDir;

uniform int u_cube;
uniform sampler2D u_cubeTexture;

void main()
{
	vec4 color = texture2D(u_cubeTexture, v_texCoord);
	if(u_cube == 0)
	{	color = color * 7.0 * vec4(1.0, 0.2, 0.0, 1.0);
	}

	float ambientFactor = 0.2;
	
	gl_FragColor = color*ambientFactor;

	if(u_cube != 0){
		vec3 direction = vec3(1,-1,-1);
		direction = normalize(direction);
		float diffFactor = 0.8*max(dot(v_normal,v_lightDir), 0.0);
		gl_FragColor += color*diffFactor;
	}		
}