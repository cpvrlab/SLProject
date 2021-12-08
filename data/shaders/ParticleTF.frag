//#############################################################################
//  File:      ParticleTD.frag
//  Purpose:   Simple GLSL fragment program for particle system
//  Date:      October 2021
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in       vec4      v_particleColor;     // interpolated color from the vertex shader

out     vec4     o_fragColor;       // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
    // Just set the interpolated color from the vertex shader
   o_fragColor = v_particleColor;
}
//-----------------------------------------------------------------------------