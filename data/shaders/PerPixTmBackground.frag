//#############################################################################
//  File:      PerPixTmBackground.frag
//  Purpose:   GLSL fragment shader for background texture mapping
//  Author:    Marcus Hudritsch
//  Date:      September 2020
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
uniform float u_bgWidth;       // background width
uniform float u_bgHeight;      // background height
uniform float u_bgLeft;        // background left
uniform float u_bgBottom;      // background bottom

uniform sampler2D u_matTexture0;      // Color map

out     vec4 o_fragColor;        // output fragment color
//-----------------------------------------------------------------------------
void main()
{
    float x = (gl_FragCoord.x - u_bgLeft) / u_bgWidth;
    float y = (gl_FragCoord.y - u_bgBottom) / u_bgHeight;

    o_fragColor = texture(u_matTexture0, vec2(x, y));
}
//-----------------------------------------------------------------------------
