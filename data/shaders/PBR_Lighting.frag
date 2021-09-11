//#############################################################################
//  File:      PBR_Lighting.frag
//  Purpose:   GLSL fragment shader for Cook-Torrance physical based rendering
//             including diffuse irradiance and specular IBL. Based on the
//             physically based rendering (PBR) tutorial with GLSL by Joey de
//             Vries on https://learnopengl.com/#!PBR/Theory
//             adapted from PerPixCookTorrance.frag by Marcus Hudritsch
//  Date:      April 2018
//  Authors:   Carlos Arauz, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces #Lights by SLVLights.size()
#pragma define NUM_LIGHTS #Lights
//-----------------------------------------------------------------------------
in      vec3    v_P_VS; // Interpol. point of illumination in view space (VS)
in      vec3    v_N_VS; // Interpol. normal at v_P_VS in view space
in      vec3    v_R_OS; // Interpol. reflected ray in object space

uniform bool    u_lightIsOn[NUM_LIGHTS];    // flag if light is on
uniform vec4    u_lightPosVS[NUM_LIGHTS];   // position of light in view space
uniform vec4    u_lightDiff[NUM_LIGHTS];    // diffuse light intensity (Id)
uniform vec4    u_lightSpec[NUM_LIGHTS];    // specular light intensity (Is)
uniform vec3    u_lightSpotDir[NUM_LIGHTS]; // spot direction in view space
uniform float   u_lightSpotDeg[NUM_LIGHTS]; // spot cutoff angle 1-180 degrees
uniform float   u_lightSpotCos[NUM_LIGHTS]; // cosine of spot cutoff angle
uniform float   u_lightSpotExp[NUM_LIGHTS]; // spot exponent
uniform float   u_oneOverGamma;             // 1.0f / Gamma correction value

uniform samplerCube u_skyIrradianceCubemap; // PBR skybox irradiance light
uniform samplerCube u_skyRoughnessCubemap;  // PBR skybox cubemap for rough reflections
uniform sampler2D   u_skyBrdfLutTexture;    // PBR lighting lookup table for BRDF
uniform float       u_skyExposure;          // PBR skybox exposure

uniform vec4        u_matDiff;      // diffuse color reflection coefficient (kd)
uniform float       u_matRough;     // Cook-Torrance material roughness 0-1
uniform float       u_matMetal;     // Cook-Torrance material metallic 0-1

uniform int         u_camProjection;    // type of stereo
uniform int         u_camStereoEye;     // -1=left, 0=center, 1=right
uniform mat3        u_camStereoColors;  // color filter matrix
uniform bool        u_camFogIsOn;       // flag if fog is on
uniform int         u_camFogMode;       // 0=LINEAR, 1=EXP, 2=EXP2
uniform float       u_camFogDensity;    // fog density value
uniform float       u_camFogStart;      // fog start distance
uniform float       u_camFogEnd;        // fog end distance
uniform vec4        u_camFogColor;      // fog color (usually the background)

out     vec4        o_fragColor;        // output fragment color
//-----------------------------------------------------------------------------
const   float       AO = 1.0;           // Constant ambient occlusion factor
const   float       PI = 3.14159265359;
//-----------------------------------------------------------------------------
#pragma include "lightingCookTorrance.glsl"
#pragma include "fogBlend.glsl"
#pragma include "doStereoSeparation.glsl"
//-----------------------------------------------------------------------------
void main()
{
    vec3 N = normalize(v_N_VS);     // A varying normal has not anymore unit length
    vec3 E = normalize(-v_P_VS);    // Vector from p to the eye (viewer)

    // Init Fresnel reflection at 90 deg. (0 to N)
    vec3 F0 = vec3(0.04);           
    F0 = mix(F0, u_matDiff.rgb, u_matMetal);

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
                                        u_matDiff.rgb,
                                        u_matMetal,
                                        u_matRough, 
                                        Lo);
            }
            else
            {
                vec3 L = u_lightPosVS[i].xyz - v_P_VS;
                vec3 S = u_lightSpotDir[i]; // normalized spot direction in VS
                pointLightCookTorrance( i, N, E, L, S, F0,
                                        u_matDiff.rgb,
                                        u_matMetal,
                                        u_matRough, 
                                        Lo);
            }
        }
    }
    
    // Build diffuse reflection for environment light map
    vec3 F = fresnelSchlickRoughness(max(dot(N, E), 0.0), F0, u_matRough);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - u_matMetal;
    vec3 irradiance = texture(u_skyIrradianceCubemap, N).rgb;
    vec3 diffuse    = kD * irradiance * u_matDiff.rgb;
    
    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(u_skyRoughnessCubemap, v_R_OS, u_matRough * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(u_skyBrdfLutTexture, vec2(max(dot(N, E), 0.0), u_matRough)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambient = (diffuse + specular) * AO;
    
    vec3 color = ambient + Lo;
    
    // Exposure tone mapping
    float skyExposure = 1.0;
    vec3 mapped = vec3(1.0) - exp(-color * u_skyExposure);
    o_fragColor = vec4(mapped, 1.0);

    // Apply fog by blending over distance
    if (u_camFogIsOn)
        o_fragColor = fogBlend(v_P_VS, o_fragColor);

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));

    // Apply stereo eye separation
    if (u_camProjection > 1)
        doStereoSeparation();
}
//-----------------------------------------------------------------------------
