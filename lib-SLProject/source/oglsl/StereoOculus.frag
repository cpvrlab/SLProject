//#############################################################################
//  File:      StereoOculus.frag
//  Purpose:   Oculus Rift Distortion Shader
//  Author:    Marc Wacker, Roman Kühne
//  Date:      November 2013
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision highp float;
#endif

uniform sampler2D	u_sceneBuffer;
uniform float		u_lensCenterOffset;
uniform vec4		u_hmdWarpParam;
uniform vec2		u_scale;
uniform vec2		u_scaleIn;

varying vec2		v_texCoord;

vec2 lensCenter = vec2(0, 0.5);
vec2 screenCenter = vec2(0.25, 0.5);

// Scales input texture coordinates for distortion.
vec2 hmdWarp(vec2 in01)
{
    vec2 theta = (in01 - lensCenter) * u_scaleIn; // Scales to [-1, 1]
    float rSq = theta.x * theta.x + theta.y * theta.y;
    float rSq2 = rSq * rSq;
    vec2 rvector = theta * (u_hmdWarpParam.x + 
                            u_hmdWarpParam.y * rSq +
                            u_hmdWarpParam.z * rSq2 +
                            u_hmdWarpParam.w * rSq2 * rSq);
    return lensCenter + u_scale * rvector;
}

void main()
{
    if(v_texCoord.x > 0.5)
    {   screenCenter.x = 0.75;
        lensCenter.x = 1.0 - u_lensCenterOffset;
        //lensCenter.x = 0.71;
    } else 
    {
        lensCenter.x = 0.0 + u_lensCenterOffset;
        //lensCenter.x = 0.29;
    }

    vec2 tc = hmdWarp(v_texCoord);
    //any tests the bool vector if any parameter is true (are we outside of the screen?)
    if(any(bvec2(clamp(tc,screenCenter-vec2(0.25,0.5), screenCenter+vec2(0.25,0.5)) - tc)))
    {
        gl_FragColor = vec4(0, 0, 0, 1);
        return;
    }

    gl_FragColor = texture2D(u_sceneBuffer, v_texCoord);
}