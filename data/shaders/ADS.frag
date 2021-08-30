//#############################################################################
//  File:      ADS.frag
//  Purpose:   GLSL fragment program for simple ADS per vertex lighting
//  Author:    Marcus Hudritsch
//  Date:      September 2011 (HS11)
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in       vec4   v_color;      // interpolated color calculated in the vertex shader

out      vec4   o_fragColor;  // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
   o_fragColor = v_color;
}
//-----------------------------------------------------------------------------