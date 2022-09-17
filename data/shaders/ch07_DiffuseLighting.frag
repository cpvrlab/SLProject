//#############################################################################
//  File:      ch07_DiffuseLighting.frag
//  Purpose:   GLSL fragment program for simple diffuse per vertex lighting
//  Date:      September 2012 (HS12)
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in       vec4   diffuseColor;   // interpolated color from the vertex shader
out      vec4   o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
   o_fragColor = diffuseColor;
}
//-----------------------------------------------------------------------------
