//#############################################################################
//  File:      SceneOculus.frag
//  Purpose:   Oculus Rift Distortion Shader
//  Author:    Marc Wacker, Roman Kuehne
//  Date:      November 2013
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

//-----------------------------------------------------------------------------
in    	vec2 		v_texCoord;
in    	vec4 		v_color;
in    	vec3 		v_normal;
in    	vec3 		v_lightDir;

uniform int 		u_cube;
uniform sampler2D 	u_cubeTexture;

out     vec4        o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
void main()
{
	vec4 color = texture(u_cubeTexture, v_texCoord);
	if(u_cube == 0)
	{	color = color * 7.0 * vec4(1.0, 0.2, 0.0, 1.0);
	}

	float ambientFactor = 0.2;
	
	o_fragColor = color*ambientFactor;

	if(u_cube != 0)
	{
		vec3 direction = vec3(1,-1,-1);
		direction = normalize(direction);
		float diffFactor = 0.8*max(dot(v_normal,v_lightDir), 0.0);
		o_fragColor += color*diffFactor;
	}		
}
//-----------------------------------------------------------------------------