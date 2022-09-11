//#############################################################################
//  File:      ch06_ColorCube.frag
//  Purpose:   Simple GLSL fragment program for constant color
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec4     v_color;       // interpolated color from the vertex shader
out     vec4     o_fragColor;   // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
    o_fragColor = v_color;      // Pass the color the fragment output color
}
//-----------------------------------------------------------------------------