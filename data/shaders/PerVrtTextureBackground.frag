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
uniform sampler2D   u_matTexture0;      // Color map
//-----------------------------------------------------------------------------
void main()
{

    vec2 texCoord = Vec2(gl_FragCoord.x / texSize.x, gl_FragCoord.y / texSize.y);
    o_fragColor = texture(u_matTexture0, texCoord);
}
//-----------------------------------------------------------------------------
