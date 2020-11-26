//#############################################################################
//  File:      PerPixBlinnTexNrmSM.frag
//  Purpose:   GLSL normal map bump mapping w. shadow mapping for max. 4 lights
//             without cube map shadow maps
//  Author:    Marcus Hudritsch
//  Date:      October 2020
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces #Lights by SLVLights.size()
#pragma define NUM_LIGHTS #Lights
//-----------------------------------------------------------------------------
in      vec3        v_P_VS;     // Interpol. point of illum. in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illum. in world space (WS)
in      vec2        v_uv1;      // Texture coordiante varying
in      vec3        v_eyeDirTS;                 // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space

uniform bool        u_lightIsOn[NUM_LIGHTS];                // flag if light is on
uniform vec4        u_lightPosVS[NUM_LIGHTS];               // position of light in view space
uniform vec4        u_lightPosWS[NUM_LIGHTS];               // position of light in world space
uniform vec4        u_lightAmbi[NUM_LIGHTS];                // ambient light intensity (Ia)
uniform vec4        u_lightDiff[NUM_LIGHTS];                // diffuse light intensity (Id)
uniform vec4        u_lightSpec[NUM_LIGHTS];                // specular light intensity (Is)
uniform vec3        u_lightSpotDir[NUM_LIGHTS];             // spot direction in view space
uniform float       u_lightSpotDeg[NUM_LIGHTS];             // spot cutoff angle 1-180 degrees
uniform float       u_lightSpotCos[NUM_LIGHTS];             // cosine of spot cutoff angle
uniform float       u_lightSpotExp[NUM_LIGHTS];             // spot exponent
uniform vec3        u_lightAtt[NUM_LIGHTS];                 // attenuation (const,linear,quadr.)
uniform bool        u_lightDoAtt[NUM_LIGHTS];               // flag if att. must be calc.
uniform vec4        u_globalAmbi;                           // Global ambient scene color
uniform float       u_oneOverGamma;                         // 1.0f / Gamma correction value
uniform mat4        u_lightSpace[NUM_LIGHTS * 6];           // projection matrices for lights
uniform bool        u_lightCreatesShadows[NUM_LIGHTS];      // flag if light creates shadows
uniform bool        u_lightDoSmoothShadows[NUM_LIGHTS];     // flag if percentage-closer filtering is enabled
uniform int         u_lightSmoothShadowLevel[NUM_LIGHTS];   // radius of area to sample for PCF
uniform float       u_lightShadowMinBias[NUM_LIGHTS];       // min. shadow bias value at 0° to N
uniform float       u_lightShadowMaxBias[NUM_LIGHTS];       // min. shadow bias value at 90° to N

uniform vec4        u_matAmbi;          // ambient color reflection coefficient (ka)
uniform vec4        u_matDiff;          // diffuse color reflection coefficient (kd)
uniform vec4        u_matSpec;          // specular color reflection coefficient (ks)
uniform vec4        u_matEmis;          // emissive color for self-shining materials
uniform float       u_matShin;          // shininess exponent
uniform sampler2D   u_matTexture0;      // Color map
uniform sampler2D   u_matTexture1;      // Normal map
uniform bool        u_matGetsShadows;   // flag if material receives shadows

uniform int         u_camProjection;    // type of stereo
uniform int         u_camStereoEye;     // -1=left, 0=center, 1=right
uniform mat3        u_camStereoColors;  // color filter matrix
uniform bool        u_camFogIsOn;       // flag if fog is on
uniform int         u_camFogMode;       // 0=LINEAR, 1=EXP, 2=EXP2
uniform float       u_camFogDensity;    // fog densitiy value
uniform float       u_camFogStart;      // fog start distance
uniform float       u_camFogEnd;        // fog end distance
uniform vec4        u_camFogColor;      // fog color (usually the background)

uniform sampler2D   u_shadowMap_0;      // shadow map for light 0
uniform sampler2D   u_shadowMap_1;      // shadow map for light 1
uniform sampler2D   u_shadowMap_2;      // shadow map for light 2
uniform sampler2D   u_shadowMap_3;      // shadow map for light 3

out     vec4        o_fragColor;        // output fragment color
//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces the include pragma by the file
#pragma include "lightingBlinnPhong.glsl"
#pragma include "fogBlend.glsl"
#pragma include "doStereoSeparation.glsl"
#pragma include "shadowTest4Lights.glsl"
//-----------------------------------------------------------------------------
void main()
{
    vec4 Ia = vec4(0.0); // Accumulated ambient light intensity at v_P_VS
    vec4 Id = vec4(0.0); // Accumulated diffuse light intensity at v_P_VS
    vec4 Is = vec4(0.0); // Accumulated specular light intensity at v_P_VS

    // Get normal from normal map, move from [0,1] to [-1, 1] range & normalize
    vec3 N = normalize(texture(u_matTexture1, v_uv1).rgb * 2.0 - 1.0);
    vec3 E = normalize(v_eyeDirTS);   // normalized eye direction

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-v_spotDirTS[i]);

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest4Lights(i, N, S) : 0.0;

                directLightBlinnPhong(i, N, E, S, shadow, Ia, Id, Is);
            }
            else
            {
                vec3 S = normalize(v_spotDirTS[i]); // normalized spot direction in TS
                vec3 L = v_lightDirTS[i]; // Vector from v_P to light in TS

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest4Lights(i, N, L) : 0.0;

                pointLightBlinnPhong(i, N, E, S, L, shadow, Ia, Id, Is);
            }
        }
    }

    // Sum up all the reflected color components
    o_fragColor =  u_matEmis +
                   u_globalAmbi +
                   Ia * u_matAmbi +
                   Id * u_matDiff;

    // Componentwise multiply w. texture color
    o_fragColor *= texture(u_matTexture0, v_uv1);

    // add finally the specular RGB-part
    vec4 specColor = Is * u_matSpec;
    o_fragColor.rgb += specColor.rgb;

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));

    // Apply fog by blending over distance
    if (u_camFogIsOn)
        o_fragColor = fogBlend(v_P_VS, o_fragColor);

    // Apply stereo eye separation
    if (u_camProjection > 1)
        doStereoSeparation();
}
//-----------------------------------------------------------------------------
