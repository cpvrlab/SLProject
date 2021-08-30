//#############################################################################
//  File:      Color.frag
//  Purpose:   Simple GLSL fragment program for constant color
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec4     v_color;           // interpolated color calculated in the vertex shader

uniform float    u_oneOverGamma;    // 1.0f / Gamma correction value

out     vec4     o_fragColor;       // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
    o_fragColor = v_color;

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------