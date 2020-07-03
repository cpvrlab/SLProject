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

/*
The preprocessor constant #define NUM_LIGHTS will be added at the shader
compilation time. It must be constant to be used in the for loop in main().
Therefore this number it can not be passed as a uniform variable.
*/

#ifdef GL_ES
precision highp float;
#endif
// ----------------------------------------------------------------------------
in      vec3        v_P_VS;              // Interpol. point of illum. in view space (VS)
in      vec3        v_N_VS;              // Interpol. normal at v_P_VS in view space
in      vec2        v_texCoord;          // Interpol. texture coordinate in tex. space

uniform bool        u_lightIsOn[NUM_LIGHTS];     // flag if light is on
uniform vec4        u_lightPosVS[NUM_LIGHTS];    // position of light in view space
uniform vec4        u_lightDiffuse[NUM_LIGHTS];  // diffuse light intensity (Id)
uniform float       u_oneOverGamma;              // 1.0f / Gamma correction value

uniform sampler2D   u_matTexture0;      // Diffuse Color map (albedo)

uniform int         u_camProjection;    // type of stereo
uniform int         u_camStereoEye;     // -1=left, 0=center, 1=right
uniform mat3        u_camStereoColors;  // color filter matrix
uniform bool        u_camFogIsOn;       // flag if fog is on
uniform int         u_camFogMode;       // 0=LINEAR, 1=EXP, 2=EXP2
uniform float       u_camFogDensity;    // fog densitiy value
uniform float       u_camFogStart;      // fog start distance
uniform float       u_camFogEnd;        // fog end distance
uniform vec4        u_camFogColor;      // fog color (usually the background)

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
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
//-----------------------------------------------------------------------------
float distributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
//-----------------------------------------------------------------------------
float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
//-----------------------------------------------------------------------------
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = geometrySchlickGGX(NdotV, roughness);
    float ggx1  = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
//-----------------------------------------------------------------------------
void pointLightCookTorrance(in    int   i,         // Light number
                            in    vec3  P_VS,      // Point of illumination in VS
                            in    vec3  N,         // Normalized normal at v_P_VS
                            in    vec3  V,         // Normalized vector from v_P_VS to view in VS
                            in    vec3  F0,        // Frenel reflection at 90 deg. (0 to N)
                            in    vec3  diffuse,   // Material diffuse color
                            in    float roughness, // Material roughness
                            in    float metallic,  // Material metallic
                            inout vec3  Lo)        // reflected intensity
{
    vec3 L = u_lightPosVS[i].xyz - v_P_VS;       // Vector from v_P_VS to the light in VS
    float distance = length(L);                  // distance to light
    L /= distance;                               // normalize light vector
    vec3 H = normalize(V + L);                   // Normalized halfvector between eye and light vector
    float att = 1.0 / (distance*distance);       // quadratic light attenuation
    vec3 radiance = u_lightDiffuse[i].rgb * att; // per light radiance

     // cook-torrance brdf
     float NDF = distributionGGX(N, H, roughness);
     float G   = geometrySmith(N, V, L, roughness);
     vec3  F   = fresnelSchlick(max(dot(H, V), 0.0), F0);

     vec3 kS = F;
     vec3 kD = vec3(1.0) - kS;
     kD *= 1.0 - metallic;

     vec3  nominator   = NDF * G * F;
     float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
     vec3  specular    = nominator / denominator;

     // add to outgoing radiance Lo
     float NdotL = max(dot(N, L), 0.0);

     Lo += (kD*diffuse/PI + specular) * radiance * NdotL;
}
//-----------------------------------------------------------------------------
vec4 fogBlend(vec3 P_VS, vec4 inColor)
{
    float factor = 0.0f;
    float distance = length(P_VS);

    switch (u_camFogMode)
    {
        case 0:
        factor = (u_camFogEnd - distance) / (u_camFogEnd - u_camFogStart);
        break;
        case 1:
        factor = exp(-u_camFogDensity * distance);
        break;
        default:
        factor = exp(-u_camFogDensity * distance * u_camFogDensity * distance);
        break;
    }

    vec4 outColor = factor * inColor + (1 - factor) * u_camFogColor;
    outColor = clamp(outColor, 0.0, 1.0);
    return outColor;
}
//-----------------------------------------------------------------------------
void doStereoSeparation()
{
    // See SLProjection in SLEnum.h
    if (u_camProjection > 8) // stereoColors
    {
        // Apply color filter but keep alpha
        o_fragColor.rgb = u_camStereoColors * o_fragColor.rgb;
    }
    else if (u_camProjection == 6) // stereoLineByLine
    {
        if (mod(floor(gl_FragCoord.y), 2.0) < 0.5)// even
        {
            if (u_camStereoEye ==-1)
            discard;
        } else // odd
        {
            if (u_camStereoEye == 1)
            discard;
        }
    }
    else if (u_camProjection == 7) // stereoColByCol
    {
        if (mod(floor(gl_FragCoord.x), 2.0) < 0.5)// even
        {
            if (u_camStereoEye ==-1)
            discard;
        } else // odd
        {
            if (u_camStereoEye == 1)
            discard;
        }
    }
    else if (u_camProjection == 8) // stereoCheckerBoard
    {
        bool h = (mod(floor(gl_FragCoord.x), 2.0) < 0.5);
        bool v = (mod(floor(gl_FragCoord.y), 2.0) < 0.5);
        if (h==v)// both even or odd
        {
            if (u_camStereoEye ==-1)
                discard;
        } else // odd
        {
            if (u_camStereoEye == 1)
                discard;
        }
    }
}
//-----------------------------------------------------------------------------
void main()
{
    // Get the material parameters out of the textures
    vec3  diffuse   = pow(texture(u_matTexture0, v_texCoord).rgb, vec3(2.2));
    float metallic  = texture(u_matTexture2, v_texCoord).r;
    float roughness = texture(u_matTexture3, v_texCoord).r;

    vec3 N = getNormalFromMap();     // Get the distracted normal from map
    vec3 V = normalize(-v_P_VS);     // Vector from p to the viewer
    vec3 F0 = vec3(0.04);            // Init Frenel reflection at 90 deg. (0 to N)
    F0 = mix(F0, diffuse, metallic);
    vec3 Lo = vec3(0.0);             // Reflection from all lights into Lo

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            //if (u_lightPosVS[i].w == 0.0)
             //   directLightBlinnPhong(i, N, E, Ia, Id, Is);
            //else
            pointLightCookTorrance(i, v_P_VS, N, V, F0, diffuse, roughness, metallic, Lo);
        }
    }

    // ambient lighting (note that the next IBL tutorial will replace
    // this ambient lighting with environment lighting).
    vec3 ambient = vec3(0.03) * diffuse * AO;
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
