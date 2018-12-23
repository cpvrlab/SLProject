//#############################################################################
//  File:      Color.frag
//  Purpose:   Simple GLSL fragment program for constant color
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision mediump float;
#endif

varying vec4    v_color;        // interpolated color calculated in the vertex shader
uniform float   u_oneOverGamma; // 1.0f / Gamma correction value

void main()
{     
    gl_FragColor = v_color;

    // Apply gamma correction
    gl_FragColor.rgb = pow(gl_FragColor.rgb, vec3(u_oneOverGamma));
}
