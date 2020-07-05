//#############################################################################
//  File:      PerPixCookTex.frag
//  Purpose:   GLSL fragment shader for Cook-Torrance physical based rendering.
//             Based on the physically based rendering (PBR) tutorial with GLSL
//             from Joey de Vries on https://learnopengl.com/#!PBR/Theory
//  Author:    Marcus Hudritsch
//  Date:      July 2017
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision highp float;
#endif
//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces #Lights by SLVLights.size()
#pragma define NUM_LIGHTS #Lights
// ----------------------------------------------------------------------------
in      vec3        v_P_VS;              // Interpol. point of illum. in view space (VS)
in      vec3        v_N_VS;              // Interpol. normal at v_P_VS in view space
in      vec2        v_texCoord;          // Interpol. texture coordinate in tex. space

uniform bool        u_lightIsOn[NUM_LIGHTS];    // flag if light is on
uniform vec4        u_lightPosVS[NUM_LIGHTS];   // position of light in view space
uniform vec4        u_lightDiff[NUM_LIGHTS];    // diffuse light intensity (Id)
uniform float       u_oneOverGamma;             // 1.0f / Gamma correction value


uniform int         u_camProjection;    // type of stereo
uniform int         u_camStereoEye;     // -1=left, 0=center, 1=right
uniform mat3        u_camStereoColors;  // color filter matrix
uniform bool        u_camFogIsOn;       // flag if fog is on
uniform int         u_camFogMode;       // 0=LINEAR, 1=EXP, 2=EXP2
uniform float       u_camFogDensity;    // fog densitiy value
uniform float       u_camFogStart;      // fog start distance
uniform float       u_camFogEnd;        // fog end distance
uniform vec4        u_camFogColor;      // fog color (usually the background)

uniform sampler2D   u_matTexture0;      // Diffuse Color map (albedo)
uniform sampler2D   u_matTexture1;      // Normal map
uniform sampler2D   u_matTexture2;      // Metallic map
uniform sampler2D   u_matTexture3;      // Roughness map

out     vec4        o_fragColor;        // output fragment color
// ----------------------------------------------------------------------------
const float AO = 1.0;               // Constant ambient occlusion factor
const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
vec3 getNormalFromMap()
{
    vec3 tangentNormal = texture(u_matTexture1, v_texCoord).xyz * 2.0 - 1.0;

    vec3 Q1  = dFdx(v_P_VS);
    vec3 Q2  = dFdy(v_P_VS);
    vec2 st1 = dFdx(v_texCoord);
    vec2 st2 = dFdy(v_texCoord);

    vec3 N  =  normalize(v_N_VS);
    vec3 T  =  normalize(Q1*st2.t - Q2*st1.t);
    vec3 B  = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}
//-----------------------------------------------------------------------------
#pragma include "lightingCookTorrance.glsl"
#pragma include "fogBlend.glsl"
#pragma include "doStereoSeparation.glsl
//-----------------------------------------------------------------------------
void main()
{
    vec3 N = getNormalFromMap();    // Get the distracted normal from map
    vec3 E = normalize(-v_P_VS);    // Vector from p to the viewer
    vec3 Lo = vec3(0.0);            // Get the reflection from all lights into Lo

    // Get the material parameters out of the textures
    vec3  matDiff  = pow(texture(u_matTexture0, v_texCoord).rgb, vec3(2.2));
    float matMetal = texture(u_matTexture2, v_texCoord).r;
    float matRough = texture(u_matTexture3, v_texCoord).r;

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            vec3 L = u_lightPosVS[i].xyz - v_P_VS;
            pointLightCookTorrance(N, E, L,
                                   u_lightDiff[i].rgb,
                                   matDiff,
                                   matMetal,
                                   matRough,
                                   Lo);
        }
    }

    // ambient lighting (note that the next IBL tutorial will replace
    // this ambient lighting with environment lighting).
    vec3 ambient = vec3(0.03) * matDiff * AO;
    vec3 color = ambient + Lo;

    // HDR tonemapping
    color = color / (color + vec3(1.0));

    // Apply gamma correction
    color.rgb = pow(color.rgb, vec3(u_oneOverGamma));

    // set the fragment color with opaque alpha
    o_fragColor = vec4(color, 1.0);

    // Apply stereo eye separation
    if (u_camProjection > 1)
        doStereoSeparation();
}
//-----------------------------------------------------------------------------
