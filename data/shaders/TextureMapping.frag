//#############################################################################
//  File:      TextureMapping.frag
//  Purpose:   GLSL fragment program for simple ADS per vertex lighting with
//             texture mapping
//  Date:      February 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in       vec4      v_color;          // interpolated color from the vertex shader
in       vec2      v_texCoord;       // interpolated texture coordinate

uniform sampler2D  u_matTextureDiffuse0;  // texture map
uniform float      u_oneOverGamma;        // 1.0f / Gamma correction value

out     vec4       o_fragColor;      // output fragment color
//-----------------------------------------------------------------------------
void main()
{  
   // Just set the interpolated color from the vertex shader
   o_fragColor = v_color;

   // componentwise multiply w. texture color
   o_fragColor *= texture(u_matTextureDiffuse0, v_texCoord);

   // Apply gamma correction
   //o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
