//#############################################################################
//  File:      TextureOnlyExternal.frag
//  Purpose:   GLSL fragment shader for texture mapping of an external texture
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#extension GL_OES_EGL_image_external_essl3 : enable

precision highp float;

//-----------------------------------------------------------------------------
in      vec2      v_uv0;            // Interpol. texture coordinate

uniform samplerExternalOES sTexture;

uniform float     u_oneOverGamma;   // 1.0f / Gamma correction value

out     vec4      o_fragColor;      // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
    o_fragColor = texture(sTexture, vec2(v_uv0.x, 1.0f - v_uv0.y));

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
