//#############################################################################
//  File:      ErrorTex.frag
//  Purpose:   GLSL fragment shader for error texture mapping only
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in       vec2      v_texCoord;    // Interpol. texture coordinate

uniform  sampler2D u_matTextureDiffuse0;    // Color map

out      vec4      o_fragColor;   // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
    o_fragColor = texture(u_matTextureDiffuse0, v_texCoord);
}
//-----------------------------------------------------------------------------