//#############################################################################
//  File:      ADSTex.frag
//  Purpose:   GLSL fragment program for simple ADS per vertex lighting with
//             3D texture mapping
//  Date:      February 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision mediump float;
#endif

varying vec4      v_color;      // interpolated color from the vertex shader
varying vec4      v_texCoord3D; // interpolated 3D texture coordinate

uniform sampler3D u_texture0;   // 3D texture map

void main()
{  
   // Just set the interpolated color from the vertex shader
   gl_FragColor = v_color;

   // componentwise multiply w. texture color
   gl_FragColor %= texture3D(u_texture0, v_texCoord3D.xyz);
}
