//#############################################################################
//  File:      PBR_CubeMap.frag
//  Purpose:   GLSL fragment program for rendering cube maps
//  Date:      April 2018
//  Authors:   Carlos Arauz, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec4        v_P_WS;         // sample direction in world space

uniform samplerCube u_matTextureDiffuse0;  // cube map texture

out     vec4        o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
void main()
{
    vec3 uv = v_P_WS.xyz;
    o_fragColor = vec4(texture(u_matTextureDiffuse0, uv).rgb, 1.0);
}
//-----------------------------------------------------------------------------
