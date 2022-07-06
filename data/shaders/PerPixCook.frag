//#############################################################################
//  File:      PerPixCook.frag
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
//-----------------------------------------------------------------------------
in      vec3    v_P_VS;// Interpol. point of illumination in view space (VS)
in      vec3    v_N_VS;// Interpol. normal at v_P_VS in view space

uniform bool    u_lightIsOn[NUM_LIGHTS];     // flag if light is on
uniform vec4    u_lightPosVS[NUM_LIGHTS];    // position of light in view space
uniform vec4    u_lightAmbi[NUM_LIGHTS];     // ambient light intensity (Ia)
uniform vec4    u_lightDiff[NUM_LIGHTS];     // diffuse light intensity (Id)
uniform vec4    u_lightSpec[NUM_LIGHTS];     // specular light intensity (Is)
uniform vec3    u_lightSpotDir[NUM_LIGHTS];  // spot direction in view space
uniform float   u_lightSpotDeg[NUM_LIGHTS];  // spot cutoff angle 1-180 degrees
uniform float   u_lightSpotCos[NUM_LIGHTS];  // cosine of spot cutoff angle
uniform float   u_lightSpotExp[NUM_LIGHTS];  // spot exponent
uniform float   u_oneOverGamma;              // 1.0f / Gamma correction value

uniform vec4    u_matDiff;  // diffuse color reflection coefficient (kd)
uniform float   u_matRough; // Cook-Torrance material roughness 0-1
uniform float   u_matMetal; // Cook-Torrance material metallic 0-1

uniform int     u_camProjType;    // type of stereo
uniform int     u_camStereoEye;     // -1=left, 0=center, 1=right
uniform mat3    u_camStereoColors;  // color filter matrix
uniform bool    u_camFogIsOn;       // flag if fog is on
uniform int     u_camFogMode;       // 0=LINEAR, 1=EXP, 2=EXP2
uniform float   u_camFogDensity;    // fog density value
uniform float   u_camFogStart;      // fog start distance
uniform float   u_camFogEnd;        // fog end distance
uniform vec4    u_camFogColor;      // fog color (usually the background)

out     vec4    o_fragColor;        // output fragment color
//-----------------------------------------------------------------------------
const float AO = 1.0;// Constant ambient occlusion factor
const float PI = 3.14159265359;
//-----------------------------------------------------------------------------
#pragma include "lightingCookTorrance.glsl"
#pragma include "fogBlend.glsl"
#pragma include "doStereoSeparation.glsl"
//-----------------------------------------------------------------------------
void main()
{
    vec3 N = normalize(v_N_VS);     // A input normal has not anymore unit length
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
                                        u_matRough, Lo);
            }
            else
            {
                vec3 L = u_lightPosVS[i].xyz - v_P_VS;
                vec3 S = u_lightSpotDir[i];// normalized spot direction in VS
                pointLightCookTorrance( i, N, E, L, S, F0,
                                        u_matDiff.rgb,
                                        u_matMetal,
                                        u_matRough, Lo);
            }
        }
    }

    // ambient lighting (note that the next IBL tutorial will replace
    // this ambient lighting with environment lighting).
    vec3 ambient = vec3(0.03) * u_matDiff.rgb * AO;
    vec3 color = ambient + Lo;

    // HDR tone-mapping
    color = color / (color + vec3(1.0));
    o_fragColor = vec4(color, 1.0);

    // Apply fog by blending over distance
    if (u_camFogIsOn)
        o_fragColor = fogBlend(v_P_VS, o_fragColor);

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));

    // Apply stereo eye separation
    if (u_camProjType > 1)
        doStereoSeparation();
}
//-----------------------------------------------------------------------------
