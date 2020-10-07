//#############################################################################
//  File:      SkyBox.frag
//  Purpose:   GLSL vertex program for unlit skybox with a cube map
//  Author:    Marcus Hudritsch
//  Date:      October 2017
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec3        v_texCoord;     // Interpol. 3D texture coordinate

uniform samplerCube u_matTexture0;     // cube map texture
uniform float       u_oneOverGamma; // 1.0f / Gamma correction value

out     vec4        o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
void main()
{
    o_fragColor = texture(u_matTexture0, v_texCoord);

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
