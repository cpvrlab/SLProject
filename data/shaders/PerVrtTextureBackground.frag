//#############################################################################
//  File:      PerVrtTextureBackground.frag
//  Purpose:   GLSL fragment shader for background texture mapping
//  Author:    Marcus Hudritsch
//  Date:      September 2020
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision mediump float;
#endif
//-----------------------------------------------------------------------------
uniform int         u_camFbWidth;       // framebuffer width
uniform int         u_camFbHeight;      // framebuffer height
uniform sampler2D   u_matTexture0;      // Color map

in      vec2        v_P_SS;             // vertex position in screen space

out     vec4        o_fragColor;        // output fragment color
//-----------------------------------------------------------------------------
void main()
{
    vec2 texCoord = vec2(gl_FragCoord.x/float(u_camFbWidth),
                         gl_FragCoord.y/float(u_camFbHeight));
    o_fragColor = texture(u_matTexture0, texCoord);
}
//-----------------------------------------------------------------------------
