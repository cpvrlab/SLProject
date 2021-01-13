//#############################################################################
//  File:      PBR_CubeMap.frag
//  Purpose:   GLSL fragment program for rendering cube maps
//  Author:    Carlos Arauz
//  Date:      April 2018
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec4        v_P_VS;         // sample direction

uniform samplerCube u_matTexture0;  // cube map texture

out     vec4        o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
void main()
{
    vec3 uv = v_P_VS;
    o_fragColor = vec4(texture(u_matTexture0, uv).rgb, 1.0);
}
//-----------------------------------------------------------------------------
