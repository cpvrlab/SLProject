//#############################################################################
//  File:      SkyBox.frag
//  Purpose:   GLSL vertex program for unlit skybox with a cube map
//  Date:      October 2017
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec3        v_uv0;          // Interpol. 3D texture coordinate

uniform samplerCube u_matTextureDiffuse0;  // cube map texture
uniform float       u_oneOverGamma; // 1.0f / Gamma correction value

out     vec4        o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
void main()
{
    o_fragColor = texture(u_matTextureDiffuse0, v_uv0);

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
