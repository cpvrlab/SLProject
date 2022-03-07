//#############################################################################
//  File:      ParticleTF.frag
//  Purpose:   Simple GLSL fragment program for particle system updating this
//  shader is never used because of the transform feedback and the 
//  rasterization off.
//  Date:      December 2021
//  Authors:   Affolter Marc
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################
precision highp float;

//-----------------------------------------------------------------------------
out     vec4     o_fragColor;       // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
   o_fragColor = vec4(0,0,0,0); // Need to be here for the compilation
}
//-----------------------------------------------------------------------------