//#############################################################################
//  File:      TextureOnly3D.frag
//  Purpose:   GLSL fragment shader for 3D texture mapping only
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision mediump float;
precision mediump sampler3D;
#endif

uniform sampler3D u_texture0;       // 3D Color map
uniform float     u_oneOverGamma;   // 1.0f / Gamma correction value
varying vec4      v_texCoord3D;     // Interpol. 3D texture coordinate

void main()
{     
    gl_FragColor = texture3D(u_texture0, v_texCoord3D.xyz);

    // Apply gamma correction
    gl_FragColor.rgb = pow(gl_FragColor.rgb, vec3(u_oneOverGamma));
}
