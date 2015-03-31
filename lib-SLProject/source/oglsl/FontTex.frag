//#############################################################################
//  File:      FontTex.frag
//  Purpose:   GLSL fragment shader for textured fonts
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision mediump float;
#endif

//-----------------------------------------------------------------------------
uniform sampler2D u_texture0;    // Color map
uniform vec4      u_textColor;   // Text color
varying vec2      v_texCoord;    // Interpol. texture coordinate

void main()
{
    // Interpolated ambient & diffuse components  
    gl_FragColor = u_textColor;
   
    // componentwise multiply w. texture color
    vec4 texCol = texture2D(u_texture0, v_texCoord);
    texCol.a = texCol.r;
    texCol.r = 1.0;
    texCol.g = 1.0;
    texCol.b = 1.0;
   
    gl_FragColor *= texCol;
}