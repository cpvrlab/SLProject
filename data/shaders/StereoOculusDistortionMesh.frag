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

uniform sampler2D  u_texture;

varying   vec2  v_texCoordR;
varying   vec2  v_texCoordG;
varying   vec2  v_texCoordB;
varying   float v_vignette;
varying   float v_timeWarp;

void main()
{
    // 3 samples for fixing chromatic aberrations
    float R = texture2D(u_texture, v_texCoordR).r;
    float G = texture2D(u_texture, v_texCoordG).g;
    float B = texture2D(u_texture, v_texCoordB).b;
    gl_FragColor = (v_vignette*vec4(R,G,B,1));
    //gl_FragColor = vec4(v_vignette, v_vignette, v_vignette, 1);
    //gl_FragColor = vec4(v_timeWarp, v_timeWarp, v_timeWarp, 1);
}

/* original hlsl shader below

Texture2D Texture : register(t0);
SamplerState Linear : register(s0);

float4 main(in float4 oPosition : SV_Position, in float2 oTexCoord0 : TEXCOORD0,
            in float2 oTexCoord1 : TEXCOORD1, in float2 oTexCoord2 : TEXCOORD2,
            in float oVignette : TEXCOORD3) : SV_Target
{
    // 3 samples for fixing chromatic aberrations
    float R = Texture.Sample(Linear, oTexCoord0.xy).r;
    float G = Texture.Sample(Linear, oTexCoord1.xy).g;
    float B = Texture.Sample(Linear, oTexCoord2.xy).b;
    return (oVignette*float4(R,G,B,1));
}


*/