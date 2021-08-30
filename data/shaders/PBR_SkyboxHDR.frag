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
uniform   float         u_exposure;     // exposure for tone mapping
uniform   samplerCube   u_matTexture0;  // cube map texture

in        vec3          v_uv1;          // Interpol. 3D texture coordinate

out       vec4          o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
void main()
{
    const float gamma = 2.2;
    vec3 hdrColor = texture(u_matTexture0, v_uv1).rgb;
  
    // Exposure tone mapping
    vec3 mapped = vec3(1.0) - exp(-hdrColor * u_exposure);
    
    // Gamma correction
    mapped = pow(mapped, vec3(1.0 / gamma));
    
    o_fragColor = vec4(mapped, 1.0);
}
//-----------------------------------------------------------------------------
