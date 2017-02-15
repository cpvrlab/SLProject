//#############################################################################
//  File:      ConstColor.frag
//  Purpose:   Simple GLSL fragment program for constant color
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision mediump float;
#endif

varying vec4 v_color;   // interpolated color calculated in the vertex shader 

void main()
{     
    gl_FragColor = v_color;
}