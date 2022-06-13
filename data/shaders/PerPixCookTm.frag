//#############################################################################
//  File:      PerPixCookTm.frag
//  Purpose:   GLSL fragment shader for Cook-Torrance physical based rendering.
//             Based on the physically based rendering (PBR) tutorial with GLSL
//             from Joey de Vries on https://learnopengl.com/#!PBR/Theory
//  Date:      July 2017
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces #Lights by SLVLights.size()
#pragma define NUM_LIGHTS #Lights
// ----------------------------------------------------------------------------
in      vec3        v_P_VS;             // Interpol. point of illumination in view space (VS)
in      vec3        v_N_VS;             // Interpol. normal at v_P_VS in view space
in      vec2        v_uv0;              // Interpol. texture coordinate in tex. space

uniform bool        u_lightIsOn[NUM_LIGHTS];    // flag if light is on
uniform vec4        u_lightPosVS[NUM_LIGHTS];   // position of light in view space
uniform vec4        u_lightDiff[NUM_LIGHTS];    // diffuse light intensity (Id)
uniform vec3        u_lightSpotDir[NUM_LIGHTS]; // spot direction in view space
uniform float       u_lightSpotDeg[NUM_LIGHTS]; // spot cutoff angle 1-180 degrees
uniform float       u_lightSpotCos[NUM_LIGHTS]; // cosine of spot cutoff angle
uniform float       u_lightSpotExp[NUM_LIGHTS]; // spot exponent
uniform float       u_oneOverGamma;             // 1.0f / Gamma correction value

uniform sampler2D   u_matTextureDiffuse0;       // Diffuse Color map (albedo)
uniform sampler2D   u_matTextureNormal0;        // Normal map
uniform sampler2D   u_matTextureMetallic0;      // Metallic map
uniform sampler2D   u_matTextureRoughness0;     // Roughness map

uniform int         u_camProjType;    // type of stereo
uniform int         u_camStereoEye;     // -1=left, 0=center, 1=right
uniform mat3        u_camStereoColors;  // color filter matrix
uniform bool        u_camFogIsOn;       // flag if fog is on
uniform int         u_camFogMode;       // 0=LINEAR, 1=EXP, 2=EXP2
uniform float       u_camFogDensity;    // fog density value
uniform float       u_camFogStart;      // fog start distance
uniform float       u_camFogEnd;        // fog end distance
uniform vec4        u_camFogColor;      // fog color (usually the background)

out     vec4        o_fragColor;        // output fragment color
// ----------------------------------------------------------------------------
const float matOccl = 1.0;                // Constant ambient occlusion factor
const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
vec3 getNormalFromMap()
{
    vec3 tangentNormal = texture(u_matTextureNormal0, v_uv0).xyz * 2.0 - 1.0;

    vec3 Q1  = dFdx(v_P_VS);
    vec3 Q2  = dFdy(v_P_VS);
    vec2 st1 = dFdx(v_uv0);
    vec2 st2 = dFdy(v_uv0);

    vec3 N  =  normalize(v_N_VS);
    vec3 T  =  normalize(Q1*st2.t - Q2*st1.t);
    vec3 B  = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}
//-----------------------------------------------------------------------------
#pragma include "lightingCookTorrance.glsl"
#pragma include "fogBlend.glsl"
#pragma include "doStereoSeparation.glsl"
//-----------------------------------------------------------------------------
void main()
{
    vec3 N = getNormalFromMap();    // Get the distracted normal from map
    vec3 E = normalize(-v_P_VS);    // Vector from p to the eye (viewer)

    // Get the material parameters out of the textures
    vec3  matDiff  = pow(texture(u_matTextureDiffuse0, v_uv0).rgb, vec3(2.2));
    float matMetal = texture(u_matTextureMetallic0, v_uv0).r;
    float matRough = texture(u_matTextureRoughness0, v_uv0).r;

    // Init Fresnel reflection at 90 deg. (0 to N)
    vec3 F0 = vec3(0.04);           
    F0 = mix(F0, matDiff.rgb, matMetal);

    // Get the reflection from all lights into Lo
    vec3 Lo = vec3(0.0);  
    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-u_lightSpotDir[i].xyz);
                directLightCookTorrance(i, N, E, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough, Lo);
            }
            else
            {
                vec3 L = u_lightPosVS[i].xyz - v_P_VS;
                vec3 S = u_lightSpotDir[i];// normalized spot direction in VS
                pointLightCookTorrance( i, N, E, L, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough, Lo);
            }
        }
    }

    // ambient lighting (note that the next IBL tutorial will replace
    // this ambient lighting with environment lighting).
    vec3 ambient = vec3(0.03) * matDiff * matOccl;
    vec3 color = ambient + Lo;

    // HDR tonemapping
    color = color / (color + vec3(1.0));

    // Apply gamma correction
    color.rgb = pow(color.rgb, vec3(u_oneOverGamma));

    // set the fragment color with opaque alpha
    o_fragColor = vec4(color, 1.0);

    // Apply stereo eye separation
    if (u_camProjType > 1)
        doStereoSeparation();
}
//-----------------------------------------------------------------------------
