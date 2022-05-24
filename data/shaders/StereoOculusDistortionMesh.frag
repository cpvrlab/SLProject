//#############################################################################
//  File:      StereoOculus.frag
//  Purpose:   Oculus Rift Distortion Shader
//  Date:      November 2013
//  Authors:   Marc Wacker, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec2        v_texCoordR;
in      vec2        v_texCoordG;
in      vec2        v_texCoordB;
in      float       v_vignette;
in      float       v_timeWarp;

uniform sampler2D   u_texture;

out     vec4        o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------

void main()
{
    // 3 samples for fixing chromatic aberrations
    float R = texture(u_texture, v_texCoordR).r;
    float G = texture(u_texture, v_texCoordG).g;
    float B = texture(u_texture, v_texCoordB).b;
    o_fragColor = (v_vignette * vec4(R,G,B,1));
}
//-----------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------