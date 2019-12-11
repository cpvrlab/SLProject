//#############################################################################
//  File:      SkyBox.frag
//  Purpose:   GLSL vertex program for unlit skybox with a cube map
//  Author:    Marcus Hudritsch
//  Date:      October 2017
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision mediump float;
#endif

uniform samplerCube u_texture0;     // cube map texture
uniform float       u_oneOverGamma; // 1.0f / Gamma correction value
varying vec3        v_texCoord;     // Interpol. 3D texture coordinate

void main()
{
    gl_FragColor = textureCube(u_texture0, v_texCoord);

    // Apply gamma correction
    gl_FragColor.rgb = pow(gl_FragColor.rgb, vec3(u_oneOverGamma));
}
