//#############################################################################
//  File:      PerPixBlinnShadowMapping.frag
//  Purpose:   GLSL per pixel lighting without texturing (and Shadow mapping)
//             Parts of this shader are based on the tutorial on
//             https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping
//             by Joey de Vries.
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision mediump float;
#endif
//-----------------------------------------------------------------------------
in       vec3        v_P_VS;                   // Interpol. point of illum. in view space (VS)
in       vec3        v_P_WS;                   // Interpol. point of illum. in world space (WS)
in       vec3        v_N_VS;                   // Interpol. normal at v_P_VS in view space
in       vec2        v_texCoord;               // interpol. texture coordinate

uniform bool        u_lightIsOn[NUM_LIGHTS];           // flag if light is on
uniform vec4        u_lightPosWS[NUM_LIGHTS];          // position of light in world space
uniform vec4        u_lightPosVS[NUM_LIGHTS];          // position of light in view space
uniform vec4        u_lightAmbient[NUM_LIGHTS];        // ambient light intensity (Ia)
uniform vec4        u_lightDiffuse[NUM_LIGHTS];        // diffuse light intensity (Id)
uniform vec4        u_lightSpecular[NUM_LIGHTS];       // specular light intensity (Is)
uniform vec3        u_lightSpotDirVS[NUM_LIGHTS];      // spot direction in view space
uniform float       u_lightSpotCutoff[NUM_LIGHTS];     // spot cutoff angle 1-180 degrees
uniform float       u_lightSpotCosCut[NUM_LIGHTS];     // cosine of spot cutoff angle
uniform float       u_lightSpotExp[NUM_LIGHTS];        // spot exponent
uniform vec3        u_lightAtt[NUM_LIGHTS];            // attenuation (const,linear,quadr.)
uniform bool        u_lightDoAtt[NUM_LIGHTS];          // flag if att. must be calc.
uniform mat4        u_lightSpace[NUM_LIGHTS * 6];      // projection matrices for lights
uniform bool        u_lightCreatesShadows[NUM_LIGHTS]; // flag if light creates shadows
uniform bool        u_lightDoesPCF[NUM_LIGHTS];        // flag if percentage-closer filtering is enabled
uniform int         u_lightPCFLevel[NUM_LIGHTS];       // radius of area to sample for PCF
uniform bool        u_lightUsesCubemap[NUM_LIGHTS];    // flag if light has a cube shadow map

uniform vec4        u_globalAmbient;          // Global ambient scene color
uniform float       u_oneOverGamma;           // 1.0f / Gamma correction value

uniform vec4        u_matAmbient;             // ambient color reflection coefficient (ka)
uniform vec4        u_matDiffuse;             // diffuse color reflection coefficient (kd)
uniform vec4        u_matSpecular;            // specular color reflection coefficient (ks)
uniform vec4        u_matEmissive;            // emissive color for self-shining materials
uniform float       u_matShininess;           // shininess exponent
uniform bool        u_matGetsShadows;         // flag if material receives shadows
uniform float       u_matShadowBias;          // Bias to use to prevent shadow acne

uniform int         u_projection;             // type of stereo
uniform int         u_stereoEye;              // -1=left, 0=center, 1=right
uniform mat3        u_stereoColorFilter;      // color filter matrix

uniform sampler2D   u_shadowMap_0;            // shadow map for light 0
uniform sampler2D   u_shadowMap_1;            // shadow map for light 1
uniform sampler2D   u_shadowMap_2;            // shadow map for light 2
uniform sampler2D   u_shadowMap_3;            // shadow map for light 3
uniform sampler2D   u_shadowMap_4;            // shadow map for light 4
uniform sampler2D   u_shadowMap_5;            // shadow map for light 5
uniform sampler2D   u_shadowMap_6;            // shadow map for light 6
uniform sampler2D   u_shadowMap_7;            // shadow map for light 7

uniform samplerCube u_shadowMapCube_0;        // cubemap for light 0
uniform samplerCube u_shadowMapCube_1;        // cubemap for light 1
uniform samplerCube u_shadowMapCube_2;        // cubemap for light 2
uniform samplerCube u_shadowMapCube_3;        // cubemap for light 3
uniform samplerCube u_shadowMapCube_4;        // cubemap for light 4
uniform samplerCube u_shadowMapCube_5;        // cubemap for light 5
uniform samplerCube u_shadowMapCube_6;        // cubemap for light 6
uniform samplerCube u_shadowMapCube_7;        // cubemap for light 7

out     vec4        o_fragColor;              // output fragment color
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
float shadowTest(in int i) // Light number
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

        // Use percentage-closer filtering (PCF) for softer shadows (if enabled)
        if (!u_lightUsesCubemap[i] && u_lightDoesPCF[i])
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
            int level = u_lightPCFLevel[i];

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
                    shadow += currentDepth - u_matShadowBias > closestDepth ? 1.0 : 0.0;
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
            if (currentDepth > closestDepth + u_matShadowBias)
                shadow = 1.0;
        }

        return shadow;
    }

    return 0.0;
}
//-----------------------------------------------------------------------------
void directLightBlinnPhong(in    int  i,   // Light number
                           in    vec3 N,   // Normalized normal at P_VS
                           in    vec3 E,   // Normalized vector from P_VS to eye in VS
                           inout vec4 Ia,  // Ambient light intensity
                           inout vec4 Id,  // Diffuse light intensity
                           inout vec4 Is)  // Specular light intensity
{
    // We use the spot light direction as the light direction vector
    vec3 L = normalize(-u_lightSpotDirVS[i].xyz);

    // Calculate diffuse & specular factors
    float diffFactor = max(dot(N,L), 0.0);
    float specFactor = 0.0;
    
    if (diffFactor!=0.0)
    {
        vec3 H = normalize(L+E); // Half vector H between L and E
        specFactor = pow(max(dot(N,H), 0.0), u_matShininess);
    }

    // Accumulate directional light intesities w/o attenuation
    Ia += u_lightAmbient[i];

    // Test if the current fragment is in shadow
    float shadow = u_matGetsShadows ? shadowTest(i) : 0.0;

    // The higher the value of the variable shadow, the less light reaches the fragment
    Id += u_lightDiffuse[i] * diffFactor * (1.0 - shadow);
    Is += u_lightSpecular[i] * specFactor * (1.0 - shadow);
}
//-----------------------------------------------------------------------------
void pointLightBlinnPhong(in    int  i,      // Light number
                          in    vec3 P_VS,   // Point of illumination in VS
                          in    vec3 N,      // Normalized normal at v_P_VS
                          in    vec3 E,      // Normalized vector from v_P_VS to view in VS
                          inout vec4 Ia,     // Ambient light intensity
                          inout vec4 Id,     // Diffuse light intensity
                          inout vec4 Is)     // Specular light intensity
{
    // Vector from v_P_VS to the light in VS
    vec3 L = u_lightPosVS[i].xyz - v_P_VS;

    // Calculate attenuation over distance & normalize L
    float att = 1.0;
    if (u_lightDoAtt[i])
    {   vec3 att_dist;
        att_dist.x = 1.0;
        att_dist.z = dot(L,L);         // = distance * distance
        att_dist.y = sqrt(att_dist.z); // = distance
        att = 1.0 / dot(att_dist, u_lightAtt[i]);
        L /= att_dist.y;               // = normalize(L)
    } else L = normalize(L);

    // Normalized halfvector between the eye and the light vector
    vec3 H = normalize(E + L);

    // Calculate diffuse & specular factors
    float diffFactor = max(dot(N,L), 0.0);
    float specFactor = 0.0;
    if (diffFactor!=0.0)
        specFactor = pow(max(dot(N,H), 0.0), u_matShininess);

    // Calculate spot attenuation
    if (u_lightSpotCutoff[i] < 180.0)
    {   float spotDot; // Cosine of angle between L and spotdir
        float spotAtt; // Spot attenuation
        spotDot = dot(-L, u_lightSpotDirVS[i]);
        if (spotDot < u_lightSpotCosCut[i]) spotAtt = 0.0;
        else spotAtt = max(pow(spotDot, u_lightSpotExp[i]), 0.0);
        att *= spotAtt;
    }

    // Accumulate light intesities
    Ia += att * u_lightAmbient[i];

    // Test if the current fragment is in shadow
    float shadow = u_matGetsShadows ? shadowTest(i) : 0.0;

    // The higher the value of the variable shadow, the less light reaches the fragment
    Id += att * u_lightDiffuse[i] * diffFactor * (1.0 - shadow);
    Is += att * u_lightSpecular[i] * specFactor * (1.0 - shadow);
}
//-----------------------------------------------------------------------------
void main()
{
    vec4 Ia, Id, Is;        // Accumulated light intensities at v_P_VS
    Ia = vec4(0.0);         // Ambient light intensity
    Id = vec4(0.0);         // Diffuse light intensity
    Is = vec4(0.0);         // Specular light intensity

    vec3 N = normalize(v_N_VS);  // A input normal has not anymore unit length
    vec3 E = normalize(-v_P_VS); // Vector from p to the eye

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
                directLightBlinnPhong(i, N, E, Ia, Id, Is);
            else
                pointLightBlinnPhong(i, v_P_VS, N, E, Ia, Id, Is);
        }
    }

    // Sum up all the reflected color components
    o_fragColor =  u_globalAmbient +
                    u_matEmissive +
                    Ia * u_matAmbient +
                    Id * u_matDiffuse +
                    Is * u_matSpecular;

    // For correct alpha blending overwrite alpha component
    o_fragColor.a = u_matDiffuse.a;

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));

    // Apply stereo eye separation
    if (u_projection > 1)
    {   if (u_projection > 7) // stereoColor??
        {   // Apply color filter but keep alpha
            o_fragColor.rgb = u_stereoColorFilter * o_fragColor.rgb;
        }
        else if (u_projection == 5) // stereoLineByLine
        {   if (mod(floor(gl_FragCoord.y), 2.0) < 0.5) // even
            {  if (u_stereoEye ==-1) discard;
            } else // odd
            {  if (u_stereoEye == 1) discard;
            }
        }
        else if (u_projection == 6) // stereoColByCol
        {   if (mod(floor(gl_FragCoord.x), 2.0) < 0.5) // even
            {  if (u_stereoEye ==-1) discard;
            } else // odd
            {  if (u_stereoEye == 1) discard;
            }
        }
        else if (u_projection == 7) // stereoCheckerBoard
        {   bool h = (mod(floor(gl_FragCoord.x), 2.0) < 0.5);
            bool v = (mod(floor(gl_FragCoord.y), 2.0) < 0.5);
            if (h==v) // both even or odd
            {  if (u_stereoEye ==-1) discard;
            } else // odd
            {  if (u_stereoEye == 1) discard;
            }
        }
    }
}
//-----------------------------------------------------------------------------
