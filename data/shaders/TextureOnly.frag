//#############################################################################
//  File:      TextureOnly.frag
//  Purpose:   GLSL fragment shader for texture mapping only
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision mediump float;
#endif
//-----------------------------------------------------------------------------
in      vec2      v_texCoord;       // Interpol. texture coordinate

uniform sampler2D u_texture0;       // Color map
uniform float     u_oneOverGamma;   // 1.0f / Gamma correction value

out     vec4      o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
    o_fragColor = texture(u_texture0, v_texCoord);

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
