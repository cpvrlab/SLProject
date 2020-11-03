//#############################################################################
//  File:      PerPixBlinnSM.frag
//  Purpose:   GLSL pixel shader for per pixel Blinn-Phong lighting with 
//             shadow mapping for max. 8 lights
//             by Joey de Vries.
//  Author:    Marcus Hudritsch
//  Date:      July 2014
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
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space

uniform bool        u_lightIsOn[NUM_LIGHTS];                // flag if light is on
uniform vec4        u_lightPosWS[NUM_LIGHTS];               // position of light in world space
uniform vec4        u_lightPosVS[NUM_LIGHTS];               // position of light in view space
uniform vec4        u_lightAmbi[NUM_LIGHTS];                // ambient light intensity (Ia)
uniform vec4        u_lightDiff[NUM_LIGHTS];                // diffuse light intensity (Id)
uniform vec4        u_lightSpec[NUM_LIGHTS];                // specular light intensity (Is)
uniform vec3        u_lightSpotDir[NUM_LIGHTS];             // spot direction in view space
uniform float       u_lightSpotDeg[NUM_LIGHTS];             // spot cutoff angle 1-180 degrees
uniform float       u_lightSpotCos[NUM_LIGHTS];             // cosine of spot cutoff angle
uniform float       u_lightSpotExp[NUM_LIGHTS];             // spot exponent
uniform vec3        u_lightAtt[NUM_LIGHTS];                 // attenuation (const,linear,quadr.)
uniform bool        u_lightDoAtt[NUM_LIGHTS];               // flag if att. must be calc.
uniform mat4        u_lightSpace[NUM_LIGHTS * 6];           // projection matrices for lights
uniform bool        u_lightCreatesShadows[NUM_LIGHTS];      // flag if light creates shadows
uniform bool        u_lightDoSmoothShadows[NUM_LIGHTS];     // flag if percentage-closer filtering is enabled
uniform int         u_lightSmoothShadowLevel[NUM_LIGHTS];   // radius of area to sample for PCF
uniform float       u_lightShadowMinBias[NUM_LIGHTS];       // min. shadow bias value at 0° to N
uniform float       u_lightShadowMaxBias[NUM_LIGHTS];       // min. shadow bias value at 90° to N
uniform bool        u_lightUsesCubemap[NUM_LIGHTS];         // flag if light has a cube shadow map

uniform vec4        u_globalAmbi;       // Global ambient scene color
uniform float       u_oneOverGamma;     // 1.0f / Gamma correction
uniform vec4        u_matAmbi;          // ambient color reflection coefficient (ka)
uniform vec4        u_matDiff;          // diffuse color reflection coefficient (kd)
uniform vec4        u_matSpec;          // specular color reflection coefficient (ks)
uniform vec4        u_matEmis;          // emissive color for self-shining materials
uniform float       u_matShin;          // shininess exponent
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
uniform sampler2D   u_shadowMap_4;      // shadow map for light 4
uniform sampler2D   u_shadowMap_5;      // shadow map for light 5
uniform sampler2D   u_shadowMap_6;      // shadow map for light 6
uniform sampler2D   u_shadowMap_7;      // shadow map for light 7

uniform samplerCube u_shadowMapCube_0;  // cubemap for light 0
uniform samplerCube u_shadowMapCube_1;  // cubemap for light 1
uniform samplerCube u_shadowMapCube_2;  // cubemap for light 2
uniform samplerCube u_shadowMapCube_3;  // cubemap for light 3
uniform samplerCube u_shadowMapCube_4;  // cubemap for light 4
uniform samplerCube u_shadowMapCube_5;  // cubemap for light 5
uniform samplerCube u_shadowMapCube_6;  // cubemap for light 6
uniform samplerCube u_shadowMapCube_7;  // cubemap for light 7

out     vec4        o_fragColor;        // output fragment color
//-----------------------------------------------------------------------------
//! SLGLShader::preprocessPragmas replaces the include pragma by the file
#pragma include "lightingBlinnPhong.glsl"
#pragma include "fogBlend.glsl"
#pragma include "doStereoSeparation.glsl"
//-----------------------------------------------------------------------------
int vectorToFace(vec3 vec) // Vector to process
{
    vec3 absVec = abs(vec);
    if (absVec.x > absVec.y && absVec.x > absVec.z)
    return vec.x > 0.0 ? 0 : 1;
    else if (absVec.y > absVec.x && absVec.y > absVec.z)
    return vec.y > 0.0 ? 2 : 3;
    else
    return vec.z > 0.0 ? 4 : 5;
}
//-----------------------------------------------------------------------------
//! Shadow text function for upto 8 lights
float shadowTest(in int i, in vec3 N, in vec3 lightDir)
{
    if (u_lightCreatesShadows[i])
    {
        // Calculate position in light space
        mat4 lightSpace;
        vec3 lightToFragment = v_P_WS - u_lightPosWS[i].xyz;

        if (u_lightUsesCubemap[i])
        lightSpace = u_lightSpace[i * 6 + vectorToFace(lightToFragment)];
        else
        lightSpace = u_lightSpace[i * 6];

        vec4 lightSpacePosition = lightSpace * vec4(v_P_WS, 1.0);

        // Normalize lightSpacePosition
        vec3 projCoords = lightSpacePosition.xyz / lightSpacePosition.w;

        // Convert to texture coordinates
        projCoords = projCoords * 0.5 + 0.5;

        float currentDepth = projCoords.z;

        // Look up depth from shadow map
        float shadow = 0.0;
        float closestDepth;

        // calculate bias between min. and max. bias depending on the angle between N and lightDir
        float bias = max(u_lightShadowMaxBias[i] * (1.0 - dot(N, lightDir)), u_lightShadowMinBias[i]);

        // Use percentage-closer filtering (PCF) for softer shadows (if enabled)
        if (!u_lightUsesCubemap[i] && u_lightDoSmoothShadows[i])
        {
            vec2 texelSize;
            if (i == 0) texelSize = 1.0 / vec2(textureSize(u_shadowMap_0, 0));
            if (i == 1) texelSize = 1.0 / vec2(textureSize(u_shadowMap_1, 0));
            if (i == 2) texelSize = 1.0 / vec2(textureSize(u_shadowMap_2, 0));
            if (i == 3) texelSize = 1.0 / vec2(textureSize(u_shadowMap_3, 0));
            if (i == 4) texelSize = 1.0 / vec2(textureSize(u_shadowMap_4, 0));
            if (i == 5) texelSize = 1.0 / vec2(textureSize(u_shadowMap_5, 0));
            if (i == 6) texelSize = 1.0 / vec2(textureSize(u_shadowMap_6, 0));
            if (i == 7) texelSize = 1.0 / vec2(textureSize(u_shadowMap_7, 0));
            int level = u_lightSmoothShadowLevel[i];

            for (int x = -level; x <= level; ++x)
            {
                for (int y = -level; y <= level; ++y)
                {
                    if (i == 0) closestDepth = texture(u_shadowMap_0, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 1) closestDepth = texture(u_shadowMap_1, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 2) closestDepth = texture(u_shadowMap_2, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 3) closestDepth = texture(u_shadowMap_3, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 4) closestDepth = texture(u_shadowMap_4, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 5) closestDepth = texture(u_shadowMap_5, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 6) closestDepth = texture(u_shadowMap_6, projCoords.xy + vec2(x, y) * texelSize).r;
                    if (i == 7) closestDepth = texture(u_shadowMap_7, projCoords.xy + vec2(x, y) * texelSize).r;
                    shadow += currentDepth - bias > closestDepth ? 1.0 : 0.0;
                }
            }
            shadow /= pow(1.0 + 2.0 * float(level), 2.0);
        }
        else
        {
            if (u_lightUsesCubemap[i])
            {
                if (i == 0) closestDepth = texture(u_shadowMapCube_0, lightToFragment).r;
                if (i == 1) closestDepth = texture(u_shadowMapCube_1, lightToFragment).r;
                if (i == 2) closestDepth = texture(u_shadowMapCube_2, lightToFragment).r;
                if (i == 3) closestDepth = texture(u_shadowMapCube_3, lightToFragment).r;
                if (i == 4) closestDepth = texture(u_shadowMapCube_4, lightToFragment).r;
                if (i == 5) closestDepth = texture(u_shadowMapCube_5, lightToFragment).r;
                if (i == 6) closestDepth = texture(u_shadowMapCube_6, lightToFragment).r;
                if (i == 7) closestDepth = texture(u_shadowMapCube_7, lightToFragment).r;
            }
            else
            {
                if (i == 0) closestDepth = texture(u_shadowMap_0, projCoords.xy).r;
                if (i == 1) closestDepth = texture(u_shadowMap_1, projCoords.xy).r;
                if (i == 2) closestDepth = texture(u_shadowMap_2, projCoords.xy).r;
                if (i == 3) closestDepth = texture(u_shadowMap_3, projCoords.xy).r;
                if (i == 4) closestDepth = texture(u_shadowMap_4, projCoords.xy).r;
                if (i == 5) closestDepth = texture(u_shadowMap_5, projCoords.xy).r;
                if (i == 6) closestDepth = texture(u_shadowMap_6, projCoords.xy).r;
                if (i == 7) closestDepth = texture(u_shadowMap_7, projCoords.xy).r;
            }

            // The fragment is in shadow if the light doesn't "see" it
            if (currentDepth > closestDepth + bias)
                shadow = 1.0;
        }

        return shadow;
    }

    return 0.0;
}
//-----------------------------------------------------------------------------
void main()
{
    vec4 Ia = vec4(0.0); // Accumulated ambient light intensity at v_P_VS
    vec4 Id = vec4(0.0); // Accumulated diffuse light intensity at v_P_VS
    vec4 Is = vec4(0.0); // Accumulated specular light intensity at v_P_VS

    vec3 N = normalize(v_N_VS);  // A input normal has not anymore unit length
    vec3 E = normalize(-v_P_VS); // Vector from p to the eye

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-u_lightSpotDir[i].xyz);

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, S) : 0.0;

                directLightBlinnPhong(i, N, E, S, shadow, Ia, Id, Is);
            }
            else
            {
                vec3 S = u_lightSpotDir[i]; // normalized spot direction in VS
                vec3 L = u_lightPosVS[i].xyz - v_P_VS; // Vector from v_P to light in VS
                
                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, L) : 0.0;

                pointLightBlinnPhong(i, N, E, S, L, shadow, Ia, Id, Is);
            }
        }
    }

    // Sum up all the reflected color components
    o_fragColor =  u_globalAmbi +
                    u_matEmis +
                    Ia * u_matAmbi +
                    Id * u_matDiff +
                    Is * u_matSpec;

    // For correct alpha blending overwrite alpha component
    o_fragColor.a = u_matDiff.a;

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
