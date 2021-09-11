//#############################################################################
//  File:      PBR_SkyboxHDR.frag
//  Purpose:   GLSL fragment program for HDR skybox with a cube map
//  Date:      April 2018
//  Authors:   Carlos Arauz. Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
uniform   samplerCube   u_matTextureEnvCubemap0;    // cube map texture
uniform   float         u_skyExposure;              // skybox exposure value

in        vec3          v_uv1;          // Interpol. 3D texture coordinate

out       vec4          o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
void main()
{
    const float gamma = 2.2;
    vec3 envColor = texture(u_matTextureEnvCubemap0, v_uv1).rgb;
  
    // Exposure tone mapping
    vec3 mapped = vec3(1.0) - exp(-envColor * u_skyExposure);
    
    // Gamma correction
    mapped = pow(mapped, vec3(1.0 / gamma));
    
    o_fragColor = vec4(mapped, 1.0);
}
//-----------------------------------------------------------------------------
