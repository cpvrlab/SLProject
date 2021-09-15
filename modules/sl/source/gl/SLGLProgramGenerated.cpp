//#############################################################################
//  File:      SLGLProgramGenerated.cpp
//  Date:      December 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLAssetManager.h>
#include <SLGLProgramManager.h>
#include <SLGLProgramGenerated.h>
#include <SLGLShader.h>
#include <SLCamera.h>
#include <SLLight.h>
#include <SLGLDepthBuffer.h>

using std::string;
using std::to_string;

///////////////////////////////
// Const. GLSL code snippets //
///////////////////////////////

//-----------------------------------------------------------------------------
string SLGLProgramGenerated::generatedShaderPath;
//-----------------------------------------------------------------------------
const string vertInputs_a_pn      = R"(
layout (location = 0) in vec4  a_position;  // Vertex position attribute
layout (location = 1) in vec3  a_normal;    // Vertex normal attribute)";
const string vertInputs_a_uv1     = R"(
layout (location = 2) in vec2  a_uv1;       // Vertex tex.coord. 1 for diffuse color)";
const string vertInputs_a_uv2     = R"(
layout (location = 3) in vec2  a_uv2;       // Vertex tex.coord. 2 for AO)";
const string vertInputs_a_tangent = R"(
layout (location = 5) in vec4  a_tangent;   // Vertex tangent attribute
)";
//-----------------------------------------------------------------------------
const string vertInputs_u_matrices_all = R"(
uniform mat3  u_nMatrix;    // normal matrix=transpose(inverse(mv))
uniform mat4  u_mMatrix;    // model matrix
uniform mat4  u_mvMatrix;   // modelview matrix
uniform mat4  u_mvpMatrix;  // = projection * modelView
)";
const string vertInputs_u_matrix_invMv = R"(
uniform mat4  u_invMvMatrix;// inverse modelview matrix
)";
//-----------------------------------------------------------------------------
const string vertInputs_u_lightNm = R"(
uniform vec4  u_lightPosVS[NUM_LIGHTS];     // position of light in view space
uniform vec3  u_lightSpotDir[NUM_LIGHTS];   // spot direction in view space
uniform float u_lightSpotDeg[NUM_LIGHTS];   // spot cutoff angle 1-180 degrees
)";
//-----------------------------------------------------------------------------
const string vertOutputs_v_P_VS    = R"(
out     vec3  v_P_VS;                   // Point of illumination in view space (VS))";
const string vertOutputs_v_P_WS    = R"(
out     vec3  v_P_WS;                   // Point of illumination in world space (WS))";
const string vertOutputs_v_N_VS    = R"(
out     vec3  v_N_VS;                   // Normal at P_VS in view space (VS))";
const string vertOutputs_v_R_OS    = R"(
out     vec3  v_R_OS;                   // Reflection vector in object space (WS))";
const string vertOutputs_v_uv1     = R"(
out     vec2  v_uv1;                    // Texture coordinate 1 output)";
const string vertOutputs_v_uv2     = R"(
out     vec2  v_uv2;                    // Texture coordinate 1 output)";
const string vertOutputs_v_lightNm = R"(
out     vec3  v_eyeDirTS;               // Vector to the eye in tangent space
out     vec3  v_lightDirTS[NUM_LIGHTS]; // Vector to the light 0 in tangent space
out     vec3  v_spotDirTS[NUM_LIGHTS];  // Spot direction in tangent space
)";
//-----------------------------------------------------------------------------
const string vertMain_Begin         = R"(

void main()
{)";
const string vertMain_v_P_VS        = R"(
    v_P_VS = vec3(u_mvMatrix *  a_position); // vertex position in view space)";
const string vertMain_v_P_WS_Sm     = R"(
    v_P_WS = vec3(u_mMatrix * a_position);   // vertex position in world space)";
const string vertMain_v_N_VS        = R"(
    v_N_VS = vec3(u_nMatrix * a_normal);     // vertex normal in view space)";
const string vertMain_v_R_OS        = R"(
    vec3 I = normalize(v_P_VS);
    vec3 N = normalize(v_N_VS);
    v_R_OS =  mat3(u_invMvMatrix) * reflect(I, v_N_VS); // = I - 2.0*dot(N,I)*N;)";
const string vertMain_v_uv1         = R"(
    v_uv1 = a_uv1;  // pass diffuse color tex.coord. 1 for interpolation)";
const string vertMainBlinn_v_uv2_Ao = R"(
    v_uv2 = a_uv2;  // pass ambient occlusion tex.coord. 2 for interpolation)";
const string vertMain_TBN_Nm        = R"(

    // Building the matrix Eye Space -> Tangent Space
    // See the math behind at: http://www.terathon.com/code/tangent.html
    vec3 n = normalize(u_nMatrix * a_normal);
    vec3 t = normalize(u_nMatrix * a_tangent.xyz);
    vec3 b = cross(n, t) * a_tangent.w; // bitangent w. corrected handedness
    mat3 TBN = mat3(t,b,n);

    // Transform vector to the eye into tangent space
    v_eyeDirTS = -v_P_VS;  // eye vector in view space
    v_eyeDirTS *= TBN;

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        // Transform spot direction into tangent space
        v_spotDirTS[i] = u_lightSpotDir[i];
        v_spotDirTS[i]  *= TBN;

        // Transform vector to the light 0 into tangent space
        vec3 L = u_lightPosVS[i].xyz - v_P_VS;
        v_lightDirTS[i] = L;
        v_lightDirTS[i] *= TBN;
    }
)";
const string vertMain_EndAll        = R"(

    // pass the vertex w. the fix-function transform
    gl_Position = u_mvpMatrix * a_position;
}
)";
//-----------------------------------------------------------------------------
const string fragInputs_u_lightAll       = R"(
uniform bool        u_lightIsOn[NUM_LIGHTS];                // flag if light is on
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
uniform vec4        u_globalAmbi;                           // Global ambient scene color
uniform float       u_oneOverGamma;                         // 1.0f / Gamma correction value
)";
const string fragInputs_u_matBlinnAll    = R"(
uniform vec4        u_matAmbi;          // ambient color reflection coefficient (ka)
uniform vec4        u_matDiff;          // diffuse color reflection coefficient (kd)
uniform vec4        u_matSpec;          // specular color reflection coefficient (ks)
uniform vec4        u_matEmis;          // emissive color for self-shining materials
uniform float       u_matShin;          // shininess exponent
)";
const string fragInputs_u_matAmbi        = R"(
uniform vec4        u_matAmbi;              // ambient color reflection coefficient (ka)
uniform vec4        u_matDiff;              // diffuse color reflection coefficient (kd)
uniform float       u_matRough;             // specular color reflection coefficient (ks)
uniform float       u_matMetal;             // emissive color for self-shining materials
)";
const string fragInputs_u_matCookAll     = R"(
uniform vec4        u_matAmbi;              // ambient color reflection coefficient (ka)
uniform vec4        u_matDiff;              // diffuse color reflection coefficient (kd)
uniform float       u_matRough;             // specular color reflection coefficient (ks)
uniform float       u_matMetal;             // emissive color for self-shining materials
)";
const string fragInputs_u_matCookEnvMaps = R"(
uniform samplerCube u_skyIrradianceCubemap; // PBR skybox irradiance light
uniform samplerCube u_skyRoughnessCubemap;  // PBR skybox cubemap for rough reflections
uniform sampler2D   u_skyBrdfLutTexture;    // PBR lighting lookup table for BRDF
uniform float       u_skyExposure;          // PBR skybox exposure
)";
//-----------------------------------------------------------------------------
const string fragInputs_u_matTexDm     = R"(
uniform sampler2D   u_matTextureDiffuse0;   // diffuse color map)";
const string fragInputs_u_matTexNm     = R"(
uniform sampler2D   u_matTextureNormal0;    // normal bump map)";
const string fragInputs_u_matTexOm     = R"(
uniform sampler2D   u_matTextureAo0;        // ambient occlusion map)";
const string fragInputs_u_matTexRm     = R"(
uniform sampler2D   u_matTextureRoughness0; // PBR material roughness texture)";
const string fragInputs_u_matTexMm     = R"(
uniform sampler2D   u_matTextureMetallic0;  // PBR material metallic texture)";
const string fragInputs_u_matTexRmMm   = R"(
uniform sampler2D   u_matTextureRoughMetal0; // PBR material roughness-metallic texture)";
const string fragInputs_u_matTexOmRmMm = R"(
uniform sampler2D   u_matTextureOcclRoughMetal0; // PBR material occlusion-roughness-metalic texture)";
const string fragInputs_u_matGetsSm    = R"(
uniform bool        u_matGetsShadows;       // flag if material receives shadows
)";
/*
const string fragInputs_u_matTmSm     = R"(
uniform sampler2D   u_matTextureDiffuse0;   // diffuse color map
uniform bool        u_matGetsShadows;       // flag if material receives shadows
)";
const string fragInputs_u_matNmSm     = R"(
uniform sampler2D   u_matTextureNormal0;    // normal bump map
uniform bool        u_matGetsShadows;       // flag if material receives shadows
)";
const string fragInputs_u_matAoSm     = R"(
uniform sampler2D   u_matTextureAo0;        // ambient occlusion map
uniform bool        u_matGetsShadows;       // flag if material receives shadows
)";
const string fragInputs_u_matNmAo     = R"(
uniform sampler2D   u_matTextureNormal0;    // normal bump map
uniform sampler2D   u_matTextureAo0;        // ambient occlusion map
)";
const string fragInputs_u_matTmNmAo   = R"(
uniform sampler2D   u_matTextureDiffuse0;   // diffuse color map
uniform sampler2D   u_matTextureNormal0;    // normal bump map
uniform sampler2D   u_matTextureAo0;        // ambient occlusion map
)";
const string fragInputs_u_matTmNmSm   = R"(
uniform sampler2D   u_matTextureDiffuse0;   // diffuse color map
uniform sampler2D   u_matTextureNormal0;    // normal bump map
uniform bool        u_matGetsShadows;       // flag if material receives shadows
)";
const string fragInputs_u_matTmAoSm   = R"(
uniform sampler2D   u_matTextureDiffuse0;   // diffuse color map
uniform sampler2D   u_matTextureAo0;        // ambient occlusion map
uniform bool        u_matGetsShadows;       // flag if material receives shadows
)";
const string fragInputs_u_matTmNmAoSm = R"(
uniform sampler2D   u_matTextureDiffuse0;   // diffuse color map
uniform sampler2D   u_matTextureNormal0;    // normal bump map
uniform sampler2D   u_matTextureAo0;        // ambient occlusion map
uniform bool        u_matGetsShadows;       // flag if material receives shadows
)";
*/
//-----------------------------------------------------------------------------
const string fragInputs_u_cam = R"(
uniform int         u_camProjection;    // type of stereo
uniform int         u_camStereoEye;     // -1=left, 0=center, 1=right
uniform mat3        u_camStereoColors;  // color filter matrix
uniform bool        u_camFogIsOn;       // flag if fog is on
uniform int         u_camFogMode;       // 0=LINEAR, 1=EXP, 2=EXP2
uniform float       u_camFogDensity;    // fog density value
uniform float       u_camFogStart;      // fog start distance
uniform float       u_camFogEnd;        // fog end distance
uniform vec4        u_camFogColor;      // fog color (usually the background)
uniform float       u_camClipNear;      // camera near plane
uniform float       u_camClipFar;       // camera far plane
uniform float       u_camBkgdWidth;     // camera background width
uniform float       u_camBkgdHeight;    // camera background height
uniform float       u_camBkgdLeft;      // camera background left
uniform float       u_camBkgdBottom;    // camera background bottom
)";

//-----------------------------------------------------------------------------
const string fragOutputs_o_fragColor = R"(
out     vec4        o_fragColor;        // output fragment color
)";
//-----------------------------------------------------------------------------
const string fragFunctionsLightingBlinnPhong = R"(
void directLightBlinnPhong(in    int  i,         // Light number between 0 and NUM_LIGHTS
                           in    vec3 N,         // Normalized normal at v_P
                           in    vec3 E,         // Normalized direction at v_P to the eye
                           in    vec3 S,         // Normalized light spot direction
                           in    float shadow,   // shadow factor
                           inout vec4 Ia,        // Ambient light intensity
                           inout vec4 Id,        // Diffuse light intensity
                           inout vec4 Is)        // Specular light intensity
{
    // Calculate diffuse & specular factors
    float diffFactor = max(dot(N, S), 0.0);
    float specFactor = 0.0;

    if (diffFactor!=0.0)
    {
        vec3 H = normalize(S + E);// Half vector H between S and E
        specFactor = pow(max(dot(N, H), 0.0), u_matShin);
    }

    // accumulate directional light intensities w/o attenuation
    Ia += u_lightAmbi[i];
    Id += u_lightDiff[i] * diffFactor * (1.0 - shadow);
    Is += u_lightSpec[i] * specFactor * (1.0 - shadow);
}

void pointLightBlinnPhong( in    int   i,
                           in    vec3  N,
                           in    vec3  E,
                           in    vec3  S,
                           in    vec3  L,
                           in    float shadow,
                           inout vec4  Ia,
                           inout vec4  Id,
                           inout vec4  Is)
{
    // Calculate attenuation over distance & normalize L
    float att = 1.0;
    if (u_lightDoAtt[i])
    {
        vec3 att_dist;
        att_dist.x = 1.0;
        att_dist.z = dot(L, L);// = distance * distance
        att_dist.y = sqrt(att_dist.z);// = distance
        att = 1.0 / dot(att_dist, u_lightAtt[i]);
        L /= att_dist.y;// = normalize(L)
    }
    else
        L = normalize(L);

    // Calculate diffuse & specular factors
    vec3 H = normalize(E + L);              // Blinn's half vector is faster than Phongs reflected vector
    float diffFactor = max(dot(N, L), 0.0); // Lambertian downscale factor for diffuse reflection
    float specFactor = 0.0;
    if (diffFactor!=0.0)    // specular reflection is only possible if surface is lit from front
        specFactor = pow(max(dot(N, H), 0.0), u_matShin); // specular shininess

    // Calculate spot attenuation
    if (u_lightSpotDeg[i] < 180.0)
    {
        float spotDot;// Cosine of angle between L and spotdir
        float spotAtt;// Spot attenuation
        spotDot = dot(-L, S);
        if (spotDot < u_lightSpotCos[i])  // if outside spot cone
            spotAtt = 0.0;
        else
            spotAtt = max(pow(spotDot, u_lightSpotExp[i]), 0.0);
        att *= spotAtt;
    }

    // Accumulate light intensities
    Ia += att * u_lightAmbi[i];
    Id += att * u_lightDiff[i] * diffFactor * (1.0 - shadow);
    Is += att * u_lightSpec[i] * specFactor * (1.0 - shadow);
}
)";
//-----------------------------------------------------------------------------
const string fragFunctionsLightingCookTorrence = R"(

//-----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}
//-----------------------------------------------------------------------------
float distributionGGX(vec3 N, vec3 H, float roughness)
{
    float PI     = 3.14159265;
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
float geometrySmith(vec3 N, vec3 E, vec3 L, float roughness)
{
    float NdotV = max(dot(N, E), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = geometrySchlickGGX(NdotV, roughness);
    float ggx1  = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}
//-----------------------------------------------------------------------------
void directLightCookTorrance(in    int   i,        // Light index
                             in    vec3  N,        // Normalized normal at v_P_VS
                             in    vec3  E,        // Normalized vector from v_P to the eye
                             in    vec3  S,        // Normalized light spot direction
                             in    vec3  F0,       // Fresnel reflection at 90 deg. (0 to N)
                             in    vec3  matDiff,  // diffuse material reflection
                             in    float matMetal, // diffuse material reflection
                             in    float matRough, // diffuse material reflection
                             in    float shadow,   // shadow factor (0.0 - 1.0)
                             inout vec3  Lo)       // reflected intensity
{
    float PI = 3.14159265;
    vec3 H = normalize(E + S);  // Normalized halfvector between eye and light vector

    vec3 radiance = u_lightDiff[i].rgb;  // Per light radiance without attenuation

    // cook-torrance brdf
    float NDF = distributionGGX(N, H, matRough);
    float G   = geometrySmith(N, E, S, matRough);
    vec3  F   = fresnelSchlick(max(dot(H, E), 0.0), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - matMetal;

    vec3  nominator   = NDF * G * F;
    float denominator = 4.0 * max(dot(N, E), 0.0) * max(dot(N, S), 0.0) + 0.001;
    vec3  specular    = nominator / denominator;

    // add to outgoing radiance Lo
    float NdotL = max(dot(N, S), 0.0);

    Lo += (kD*matDiff.rgb/PI + specular) * radiance * NdotL * (1.0 - shadow);
}
//-----------------------------------------------------------------------------
void pointLightCookTorrance(in    int   i,        // Light index
                            in    vec3  N,        // Normalized normal at v_P_VS
                            in    vec3  E,        // Normalized vector from v_P to the eye
                            in    vec3  L,        // Vector from v_P to the light
                            in    vec3  S,        // Normalized light spot direction
                            in    vec3  F0,       // Fresnel reflection at 90 deg. (0 to N)
                            in    vec3  matDiff,  // diffuse material reflection
                            in    float matMetal, // diffuse material reflection
                            in    float matRough, // diffuse material reflection
                            in    float shadow,   // shadow factor (0.0 - 1.0)
                            inout vec3  Lo)       // reflected intensity
{
    float PI = 3.14159265;
    float distance = length(L); // distance to light
    L /= distance;              // normalize light vector
    float att = 1.0 / (distance*distance);  // quadratic light attenuation

    // Calculate spot attenuation
    if (u_lightSpotDeg[i] < 180.0)
    {
        float spotAtt; // Spot attenuation
        float spotDot; // Cosine of angle between L and spotdir
        spotDot = dot(-L, S);
        if (spotDot < u_lightSpotCos[i]) spotAtt = 0.0;
        else spotAtt = max(pow(spotDot, u_lightSpotExp[i]), 0.0);
        att *= spotAtt;
    }

    vec3 radiance = u_lightDiff[i].rgb * att;  // per light radiance

    // cook-torrance brdf
    vec3  H   = normalize(E + L);  // Normalized halfvector between eye and light vector
    float NDF = distributionGGX(N, H, matRough);
    float G   = geometrySmith(N, E, L, matRough);
    vec3  F   = fresnelSchlick(max(dot(H, E), 0.0), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - matMetal;

    vec3  nominator   = NDF * G * F;
    float denominator = 4.0 * max(dot(N, E), 0.0) * max(dot(N, L), 0.0) + 0.001;
    vec3  specular    = nominator / denominator;

    // add to outgoing radiance Lo
    float NdotL = max(dot(N, L), 0.0);

    Lo += (kD*matDiff.rgb/PI + specular) * radiance * NdotL * (1.0 - shadow);
}
)";
//-----------------------------------------------------------------------------
const string fragFunctionDoStereoSeparation = R"(
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
)";
//-----------------------------------------------------------------------------
const string fragFunctionFogBlend = R"(
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

    vec4 outColor = factor * inColor + (1.0 - factor) * u_camFogColor;
    outColor = clamp(outColor, 0.0, 1.0);
    return outColor;
}
)";
//-----------------------------------------------------------------------------
const string fragFunctionDoColoredShadows = R"(
void doColoredShadows(in vec3 N)
{
    const vec3 SHADOW_COLOR[6] = vec3[6](vec3(1.0, 0.0, 0.0),
                                         vec3(0.0, 1.0, 0.0),
                                         vec3(0.0, 0.0, 1.0),
                                         vec3(1.0, 1.0, 0.0),
                                         vec3(0.0, 1.0, 1.0),
                                         vec3(1.0, 0.0, 1.0));

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
                if (u_lightNumCascades[i] > 0)
                {
                    int casIndex = getCascadesDepthIndex(i, u_lightNumCascades[i]);
                    o_fragColor.rgb += shadow * SHADOW_COLOR[casIndex];
                } else
                    o_fragColor.rgb += shadow * SHADOW_COLOR[0];
            }
            else
            {
                vec3 L = u_lightPosVS[i].xyz - v_P_VS; // Vector from v_P to light in VS

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, L) : 0.0;
                o_fragColor.rgb += shadow * SHADOW_COLOR[0];
            }
        }
    }
})";
//-----------------------------------------------------------------------------
const string fragMain_Begin         = R"(

void main()
{)";
const string fragMain_0_Intensities = R"(
    vec4 Ia = vec4(0.0); // Accumulated ambient light intensity at v_P_VS
    vec4 Id = vec4(0.0); // Accumulated diffuse light intensity at v_P_VS
    vec4 Is = vec4(0.0); // Accumulated specular light intensity at v_P_VS
)";
//-----------------------------------------------------------------------------
const string fragMain_1_EN_fromVert = R"(
    vec3 E = normalize(-v_P_VS); // Interpolated vector from p to the eye
    vec3 N = normalize(v_N_VS);  // A input normal has not anymore unit length
)";
const string fragMain_1_EN_fromNm   = R"(
    vec3 E = normalize(v_eyeDirTS);   // normalized interpolated eye direction
    // Get normal from normal map, move from [0,1] to [-1, 1] range & normalize
    vec3 N = normalize(texture(u_matTextureNormal0, v_uv1).rgb * 2.0 - 1.0);
)";
//-----------------------------------------------------------------------------
const string fragMainBlinn_2_LightLoop     = R"(
    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-u_lightSpotDir[i].xyz);
                directLightBlinnPhong(i, N, E, S, 0.0, Ia, Id, Is);
            }
            else
            {
                vec3 S = u_lightSpotDir[i]; // normalized spot direction in VS
                vec3 L = u_lightPosVS[i].xyz - v_P_VS; // Vector from v_P to light in VS
                pointLightBlinnPhong(i, N, E, S, L, 0.0, Ia, Id, Is);
            }
        }
    }
)";
const string fragMainBlinn_2_LightLoopNm   = R"(
    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-v_spotDirTS[i]);
                directLightBlinnPhong(i, N, E, S, 0.0, Ia, Id, Is);
            }
            else
            {
                vec3 S = normalize(v_spotDirTS[i]); // normalized spot direction in TS
                vec3 L = v_lightDirTS[i]; // Vector from v_P to light in TS
                pointLightBlinnPhong(i, N, E, S, L, 0.0, Ia, Id, Is);
            }
        }
    }
)";
const string fragMainBlinn_2_LightLoopSm   = R"(
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
)";
const string fragMainBlinn_2_LightLoopNmSm = R"(
    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                // We use the spot light direction as the light direction vector
                vec3 S = normalize(-v_spotDirTS[i]);

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, S) : 0.0;
                directLightBlinnPhong(i, N, E, S, shadow, Ia, Id, Is);
            }
            else
            {
                vec3 S = normalize(v_spotDirTS[i]); // normalized spot direction in TS
                vec3 L = v_lightDirTS[i]; // Vector from v_P to light in TS

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, L) : 0.0;
                pointLightBlinnPhong(i, N, E, S, L, shadow, Ia, Id, Is);
            }
        }
    }
)";
const string fragMainBlinn_3_FragColor     = R"(
    // Sum up all the reflected color components
    o_fragColor =  u_matEmis +
                   u_globalAmbi +
                   Ia * u_matAmbi +
                   Id * u_matDiff +
                   Is * u_matSpec;

    // For correct alpha blending overwrite alpha component
    o_fragColor.a = u_matDiff.a;
)";
const string fragMainBlinn_3_FragColorTm   = R"(
    // Sum up all the reflected color components
    o_fragColor =  u_matEmis +
                   u_globalAmbi +
                   Ia * u_matAmbi +
                   Id * u_matDiff;

    // Componentwise multiply w. texture color
    o_fragColor *= texture(u_matTextureDiffuse0, v_uv1);

    // add finally the specular RGB-part
    vec4 specColor = Is * u_matSpec;
    o_fragColor.rgb += specColor.rgb;
)";
const string fragMainBlinn_3_FragColorAo   = R"(
    // Get ambient occlusion factor
    float AO = texture(u_matTextureAo0, v_uv2).r;

    // Sum up all the reflected color components
    o_fragColor =  u_matEmis +
                   u_globalAmbi +
                   Ia * u_matAmbi * AO +
                   Id * u_matDiff +
                   Is * u_matSpec;

    // For correct alpha blending overwrite alpha component
    o_fragColor.a = u_matDiff.a;
)";
const string fragMainBlinn_3_FragColorTmAo = R"(
    // Get ambient occlusion factor
    float AO = texture(u_matTextureAo0, v_uv2).r;

    // Sum up all the reflected color components
    o_fragColor =  u_matEmis +
                   u_globalAmbi +
                   Ia * u_matAmbi * AO +
                   Id * u_matDiff;

    // Componentwise multiply w. texture color
    o_fragColor *= texture(u_matTextureDiffuse0, v_uv1);

    // add finally the specular RGB-part
    vec4 specColor = Is * u_matSpec;
    o_fragColor.rgb += specColor.rgb;
)";
//-----------------------------------------------------------------------------
const string fragMain_4_ColoredShadows = R"(
    // Colorize cascaded shadows for debugging purpose
    if (u_lightsDoColoredShadows)
        doColoredShadows(N);
)";
//-----------------------------------------------------------------------------
const string fragMain_5_FogGammaStereo = R"(
    // Apply fog by blending over distance
    if (u_camFogIsOn)
        o_fragColor = fogBlend(v_P_VS, o_fragColor);

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));

    // Apply stereo eye separation
    if (u_camProjection > 1)
        doStereoSeparation();
}
)";
//-----------------------------------------------------------------------------
const string fragMainCook_1_MatFromMaterial = R"(
    vec4  matDiff  = u_matDiff;
    float matRough = u_matRough;
    float matMetal = u_matMetal;
)";
const string fragMainCook_1_MatFromDmRmMm   = R"(
    vec4  matDiff  = pow(texture(u_matTextureDiffuse0, v_uv1), vec4(2.2));
    float matRough = texture(u_matTextureRoughness0, v_uv1).r;
    float matMetal = texture(u_matTextureMetallic0, v_uv1).r;
)";
const string fragMainCook_1_MatFromDmRMm    = R"(
    vec4  matDiff  = pow(texture(u_matTextureDiffuse0, v_uv1), vec4(2.2));
    float matRough = texture(u_matTextureRoughMetallic0, v_uv1).g;
    float matMetal = texture(u_matTextureRoughMetallic0, v_uv1).b;
)";
const string fragMainCook_2_LightLoop       = R"(
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
                                        matRough,
                                        0.0,
                                        Lo);
            }
            else
            {
                vec3 L = u_lightPosVS[i].xyz - v_P_VS;
                vec3 S = u_lightSpotDir[i]; // normalized spot direction in VS
                pointLightCookTorrance( i, N, E, L, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        0.0,
                                        Lo);
            }
        }
    }
)";
const string fragMainCook_2_LightLoopNm     = R"(
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
                vec3 S = normalize(-v_spotDirTS[i]);
                directLightCookTorrance(i, N, E, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        0.0,
                                        Lo);
            }
            else
            {
                vec3 L = v_lightDirTS[i]; // Vector from v_P to light in TS
                vec3 S = normalize(-v_spotDirTS[i]);
                pointLightCookTorrance( i, N, E, L, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        0.0,
                                        Lo);
            }
        }
    }
)";
const string fragMainCook_2_LightLoopSm     = R"(
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

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, S) : 0.0;
                directLightCookTorrance(i, N, E, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        shadow,
                                        Lo);
            }
            else
            {
                vec3 L = u_lightPosVS[i].xyz - v_P_VS;
                vec3 S = u_lightSpotDir[i]; // normalized spot direction in VS

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, L) : 0.0;
                pointLightCookTorrance( i, N, E, L, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        shadow,
                                        Lo);
            }
        }
    }
)";
const string fragMainCook_2_LightLoopNmSm   = R"(
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
                vec3 S = normalize(-v_spotDirTS[i]);

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, S) : 0.0;
                directLightCookTorrance(i, N, E, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        shadow,
                                        Lo);
            }
            else
            {
                vec3 L = v_lightDirTS[i]; // Vector from v_P to light in TS
                vec3 S = normalize(-v_spotDirTS[i]);

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, L) : 0.0;
                pointLightCookTorrance( i, N, E, L, S, F0,
                                        matDiff.rgb,
                                        matMetal,
                                        matRough,
                                        shadow,
                                        Lo);
            }
        }
    }
)";
const string fragMainCook_3_FragColor       = R"(

    // ambient lighting (note that the next IBL tutorial will replace
    // this ambient lighting with environment lighting).
    vec3 ambient = vec3(0.03) * matDiff.rgb;
    vec3 color = ambient + Lo;

    // HDR tone-mapping
    color = color / (color + vec3(1.0));
    o_fragColor = vec4(color, 1.0);
)";
const string fragMainCook_3_FragColorOm     = R"(

    // ambient lighting (note that the next IBL tutorial will replace
    // this ambient lighting with environment lighting).
    float matAO    = texture(u_matTextureAo0, v_uv1).r;
    vec3 ambient = vec3(0.03) * matDiff.rgb * matAO;
    vec3 color = ambient + Lo;

    // HDR tone-mapping
    color = color / (color + vec3(1.0));
    o_fragColor = vec4(color, 1.0);
)";
const string fragMainCook_3_FragColorTm     = R"(

    // Build diffuse reflection for environment light map
    float exposureToneMapping = 1.0f;

    vec3 ambient = vec3(0.03) * matDiff.rgb;
    vec3 color = ambient + Lo;

    // HDR tonemapping
    color = color / (color + vec3(1.0));
    o_fragColor = vec4(color, 1.0);
)";
const string fragMainCook_3_FragColorDmOm   = R"(

    // Build diffuse reflection for environment light map
    float exposureToneMapping = 1.0f;
    float matAO    = texture(u_matTextureAo0, v_uv1).r;

    vec3 ambient = vec3(0.03) * matDiff.rgb + matAO;
    vec3 color = ambient + Lo;

    // HDR tonemapping
    color = color / (color + vec3(1.0));
    o_fragColor = vec4(color, 1.0);
)";
const string fragMainCook_3_FragColorEm     = R"(

    // Build diffuse reflection for environment light map
    vec3 F = fresnelSchlickRoughness(max(dot(N, E), 0.0), F0, u_matRough);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - matMetal;
    vec3 irradiance = texture(u_skyIrradianceCubemap, N).rgb;
    vec3 diffuse    = kD * irradiance * matDiff.rgb;

    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(u_skyRoughnessCubemap, v_R_OS, u_matRough * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(u_skyBrdfLutTexture, vec2(max(dot(N, E), 0.0), u_matRough)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);
    vec3 ambient = (diffuse + specular);

    vec3 color = ambient + Lo;

    // Exposure tone mapping
    vec3 mapped = vec3(1.0) - exp(-color * u_skyExposure);
    o_fragColor = vec4(mapped, 1.0);

    // For correct alpha blending overwrite alpha component
    o_fragColor.a = matDiff.a;
)";
const string fragMainCook_3_FragColorOmEm   = R"(

    // Build diffuse reflection for environment light map
    float matAO    = texture(u_matTextureAo0, v_uv1).r;
    vec3 F = fresnelSchlickRoughness(max(dot(N, E), 0.0), F0, u_matRough);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - matMetal;
    vec3 irradiance = texture(u_skyIrradianceCubemap, N).rgb;
    vec3 diffuse    = kD * irradiance * matDiff.rgb;

    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(u_skyRoughnessCubemap, v_R_OS, u_matRough * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(u_skyBrdfLutTexture, vec2(max(dot(N, E), 0.0), u_matRough)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);
    vec3 ambient = (diffuse + specular) * matAO;

    vec3 color = ambient + Lo;

    // Exposure tone mapping
    vec3 mapped = vec3(1.0) - exp(-color * u_skyExposure);
    o_fragColor = vec4(mapped, 1.0);

    // For correct alpha blending overwrite alpha component
    o_fragColor.a = matDiff.a;
)";
const string fragMainCook_3_FragColorDmEm   = R"(

    // Build diffuse reflection for environment light map
    vec3 F = fresnelSchlickRoughness(max(dot(N, E), 0.0), F0, matRough);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - matMetal;
    vec3 irradiance = texture(u_skyIrradianceCubemap, N).rgb;
    vec3 diffuse    = kD * irradiance * matDiff.rgb;

    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(u_skyRoughnessCubemap, v_R_OS, matRough * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(u_skyBrdfLutTexture, vec2(max(dot(N, E), 0.0), matRough)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);
    vec3 ambient = (kD * diffuse + specular);

    vec3 color = ambient + Lo;

    // Exposure tone mapping
    vec3 mapped = vec3(1.0) - exp(-color * u_skyExposure);
    o_fragColor = vec4(mapped, 1.0);
)";
const string fragMainCook_3_FragColorDmOmEm = R"(

    // Build diffuse reflection for environment light map
    float matAO    = texture(u_matTextureAo0, v_uv1).r;
    vec3 F = fresnelSchlickRoughness(max(dot(N, E), 0.0), F0, matRough);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - matMetal;
    vec3 irradiance = texture(u_skyIrradianceCubemap, N).rgb;
    vec3 diffuse    = kD * irradiance * matDiff.rgb;

    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(u_skyRoughnessCubemap, v_R_OS, matRough * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(u_skyBrdfLutTexture, vec2(max(dot(N, E), 0.0), matRough)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);
    vec3 ambient = (kD * diffuse + specular) * matAO;

    vec3 color = ambient + Lo;

    // Exposure tone mapping
    vec3 mapped = vec3(1.0) - exp(-color * u_skyExposure);
    o_fragColor = vec4(mapped, 1.0);
)";
//-----------------------------------------------------------------------------
const string fragMainVideoBkgd = R"(
void main()
{
    float x = (gl_FragCoord.x - u_camBkgdLeft) / u_camBkgdWidth;
    float y = (gl_FragCoord.y - u_camBkgdBottom) / u_camBkgdHeight;

    if(x < 0.0f || y < 0.0f || x > 1.0f || y > 1.0f)
        o_fragColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    else
        o_fragColor = texture(u_matTextureDiffuse0, vec2(x, y));

    vec3 N = normalize(v_N_VS);  // A input normal has not anymore unit length
    float shadow = 0.0;

    // Colorize cascaded shadows for debugging purpose
    if (u_lightsDoColoredShadows)
        doColoredShadows(N);
    else
    {
        for (int i = 0; i < NUM_LIGHTS; ++i)
        {
            if (u_lightIsOn[i])
            {
                if (u_lightPosVS[i].w == 0.0)
                {
                    // We use the spot light direction as the light direction vector
                    vec3 S = normalize(-u_lightSpotDir[i].xyz);

                    // Test if the current fragment is in shadow
                    shadow = u_matGetsShadows ? shadowTest(i, N, S) : 0.0;
                }
                else
                {
                    vec3 L = u_lightPosVS[i].xyz - v_P_VS; // Vector from v_P to light in VS

                    // Test if the current fragment is in shadow
                    shadow = u_matGetsShadows ? shadowTest(i, N, L) : 0.0;
                }
                o_fragColor = o_fragColor * min(1.0 - shadow + u_matAmbi.r, 1.0);
            }
        }
    }

)";
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//! Builds unique program name that identifies shader program
/*! See the class information for more insights of the generated name. This
 * function is used in advance of the code generation to check if the program
 * already exists in the asset manager. See SLMaterial::activate.
 * @param mat Parent material pointer
 * @param lights Pointer of vector of lights
 */
void SLGLProgramGenerated::buildProgramName(SLMaterial* mat,
                                            SLVLight*   lights,
                                            string&     programName)
{
    assert(mat && "No material pointer passed!");
    assert(lights && !lights->empty() && "No lights passed!");
    programName = "gen";

    bool matHasDm = mat->hasTextureType(TT_diffuse);
    bool matHasNm = mat->hasTextureType(TT_normal);
    bool matHasHm = mat->hasTextureType(TT_height);
    bool matHasAo = mat->hasTextureType(TT_occlusion);
    bool matHasVb = mat->hasTextureType(TT_videoBkgd);

    if (matHasVb)
        programName += "PerPixVideoBkgdDm";
    else if (mat->lightModel() == LM_BlinnPhong)
        programName += "PerPixBlinn";
    else if (mat->lightModel() == LM_CookTorrance)
        programName += "PerPixCook";
    else
        programName += "Custom";

    if (matHasDm)
        programName += "Dm";
    if (matHasNm && !matHasHm)
        programName += "Nm";
    if (matHasAo)
        programName += "Om";
    programName += "-";

    // Add letter per light type
    for (auto light : *lights)
    {
        if (light->positionWS().w == 0.0f)
        {
            if (light->doCascadedShadows())
                programName += "C" + std::to_string(light->shadowMap()->numCascades()); // Directional light with cascaded shadowmap
            else
                programName += "D"; // Directional light
        }
        else if (light->spotCutOffDEG() < 180.0f)
            programName += "S"; // Spot light
        else
            programName += "P"; // Point light
        if (light->createsShadows())
            programName += "s"; // Creates shadows
    }
}
//-----------------------------------------------------------------------------
/*! Builds the GLSL program code for the vertex and fragment shaders. The code
 * is only assembled but not compiled and linked. This happens within the
 * before the first draw call from within SLMesh::draw.
 * @param mat Parent material pointer
 * @param lights Pointer of vector of lights
 */
void SLGLProgramGenerated::buildProgramCode(SLMaterial* mat,
                                            SLVLight*   lights)
{
    if (mat->name() == "IBLMat")
    {
        std::cout << "build program code for IBLMat" << std::endl;
    }
    assert(mat && "No material pointer passed!");
    assert(!lights->empty() && "No lights passed!");
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Check what textures the material has
    bool Dm  = mat->hasTextureType(TT_diffuse);   // Texture Mapping
    bool Nm  = mat->hasTextureType(TT_normal);    // Normal Mapping
    bool Hm  = mat->hasTextureType(TT_height);    // Height Mapping
    bool Om  = mat->hasTextureType(TT_occlusion); // Ambient Occlusion Mapping
    bool Vm  = mat->hasTextureType(TT_videoBkgd); // Video Background Mapping
    bool env = mat->skybox() != nullptr;          // Environment Mapping from skybox

    // Check if any of the scene lights does shadow mapping
    bool Sm = lightsDoShadowMapping(lights);

    if (mat->lightModel() == LM_BlinnPhong)
    {
        if (Dm && Nm && Om && Sm)
            buildPerPixBlinnDmNmOmSm(lights);
        else if (Dm && Nm && Om)
            buildPerPixBlinnDmNmOm(lights);
        else if (Dm && Nm && Sm)
            buildPerPixBlinnDmNmSm(lights);
        else if (Dm && Om && Sm)
            buildPerPixBlinnDmOmSm(lights);
        else if (Om && Sm)
            buildPerPixBlinnOmSm(lights);
        else if (Nm && Sm)
            buildPerPixBlinnNmSm(lights);
        else if (Dm && Sm)
            buildPerPixBlinnDmSm(lights);
        else if (Nm && Om)
            buildPerPixBlinnNmOm(lights);
        else if (Dm && Om)
            buildPerPixBlinnDmOm(lights);
        else if (Dm && Nm)
            buildPerPixBlinnDmNm(lights);
        else if (Sm)
            buildPerPixBlinnSm(lights);
        else if (Om)
            buildPerPixBlinnOm(lights);
        else if (Nm)
            buildPerPixBlinnNm(lights);
        else if (Dm)
            buildPerPixBlinnDm(lights);
        else
            buildPerPixBlinn(lights);
    }
    else if (mat->lightModel() == LM_CookTorrance)
    {

        // buildPerPixCook(mat, lights);

        if (Dm && Om && Nm & Sm)
            buildPerPixCookDmNmOmSm(lights, env);
        else if (Dm && Om && Nm)
            buildPerPixCookDmNmOm(lights, env);
        else if (Dm && Nm && Sm)
            buildPerPixCookDmNmSm(lights, env);
        else if (Dm && Om && Sm)
            buildPerPixCookDmOmSm(lights, env);
        else if (Om && Sm)
            buildPerPixCookOmSm(lights, env);
        else if (Nm && Sm)
            buildPerPixCookNmSm(lights, env);
        else if (Dm && Sm)
            buildPerPixCookDmSm(lights, env);
        else if (Dm && Om)
            buildPerPixCookDmOm(lights, env);
        else if (Dm && Nm)
            buildPerPixCookDmNm(lights, env);
        else if (Sm)
            buildPerPixCookSm(lights, env);
        else if (Om)
            buildPerPixCookOm(lights, env);
        else if (Dm)
            buildPerPixCookDm(lights, env);
        else if (Nm)
            buildPerPixCookNm(lights, env);
        else
            buildPerPixCook(lights, env);
    }
    else if (mat->lightModel() == LM_Custom)
    {
        if (Vm && Sm)
            buildPerPixVideoBkgdSm(lights);
        else
            SL_EXIT_MSG("SLGLProgramGenerated::buildProgramCode: Unknown program for LM_Custom.");
    }
    else
        SL_EXIT_MSG("SLGLProgramGenerated::buildProgramCode: Unknown Lighting Model.");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCook(SLMaterial* mat, SLVLight* lights)
{
    assert(mat && lights);
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Check what textures the material has
    bool Dm   = mat->hasTextureType(TT_diffuse);
    bool Nm   = mat->hasTextureType(TT_normal);
    bool Hm   = mat->hasTextureType(TT_height);
    bool Rm   = mat->hasTextureType(TT_roughness);
    bool Mm   = mat->hasTextureType(TT_metallic);
    bool RMm  = mat->hasTextureType(TT_roughMetallic);
    bool Em   = mat->hasTextureType(TT_emissive);
    bool Om   = mat->hasTextureType(TT_occlusion);
    bool ORMm = mat->hasTextureType(TT_occlRoughMetallic);
    bool Vm   = mat->hasTextureType(TT_videoBkgd);
    bool Sm   = lightsDoShadowMapping(lights);
    bool env  = mat->skybox() != nullptr;

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    if (Dm) vertCode += vertInputs_a_uv1;
    if (Nm) vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    if (env) vertCode += vertOutputs_v_R_OS;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + "-new.vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());

    addCodeToShader(_shaders[1], fragCode, _name + "-new.frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCookDmNmOmSm(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
in      vec2        v_uv1;      // Texture coordinate varying
in      vec3        v_eyeDirTS; // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    if (doEnvMap)
        fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexNm;
    fragCode += fragInputs_u_matTexOm;
    fragCode += fragInputs_u_matTexRm;
    fragCode += fragInputs_u_matTexMm;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromNm;
    fragCode += fragMainCook_1_MatFromDmRmMm;
    fragCode += fragMainCook_2_LightLoopNmSm;
    if (doEnvMap)
        fragCode += fragMainCook_3_FragColorDmOmEm;
    else
        fragCode += fragMainCook_3_FragColorDmOm;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCookDmNmOm(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
in      vec2        v_uv1;      // Texture coordinate varying
in      vec3        v_eyeDirTS;                 // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
)";
    fragCode += fragInputs_u_lightAll;
    if (doEnvMap)
        fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexNm;
    fragCode += fragInputs_u_matTexRm;
    fragCode += fragInputs_u_matTexMm;
    fragCode += fragInputs_u_matTexOm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromNm;
    fragCode += fragMainCook_1_MatFromDmRmMm;
    fragCode += fragMainCook_2_LightLoopNm;
    if (doEnvMap)
        fragCode += fragMainCook_3_FragColorDmOmEm;
    else
        fragCode += fragMainCook_3_FragColorDmOm;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCookDmNmSm(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
in      vec2        v_uv1;      // Texture coordinate varying
in      vec3        v_eyeDirTS; // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    if (doEnvMap)
        fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexNm;
    fragCode += fragInputs_u_matTexRm;
    fragCode += fragInputs_u_matTexMm;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromNm;
    fragCode += fragMainCook_1_MatFromDmRmMm;
    fragCode += fragMainCook_2_LightLoopNmSm;
    if (doEnvMap)
        fragCode += fragMainCook_3_FragColorDmEm;
    else
        fragCode += fragMainCook_3_FragColorTm;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCookDmOmSm(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
in      vec2        v_uv1;      // Texture coordinate varying
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    if (doEnvMap)
        fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexOm;
    fragCode += fragInputs_u_matTexRm;
    fragCode += fragInputs_u_matTexMm;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainCook_1_MatFromDmRmMm;
    fragCode += fragMainCook_2_LightLoopSm;
    if (doEnvMap)
        fragCode += fragMainCook_3_FragColorDmOmEm;
    else
        fragCode += fragMainCook_3_FragColorDmOm;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCookOmSm(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
in      vec2        v_uv1;      // Texture coordinate varying
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    fragCode += fragInputs_u_matCookAll;
    if (doEnvMap)
        fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_matTexOm;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainCook_1_MatFromMaterial;
    fragCode += fragMainCook_2_LightLoop;
    if (doEnvMap)
        fragCode += fragMainCook_3_FragColorOmEm;
    else
        fragCode += fragMainCook_3_FragColorOm;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCookNmSm(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
in      vec2        v_uv1;      // Texture coordinate varying
in      vec3        v_eyeDirTS;                 // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    if (doEnvMap)
        fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_matTexNm;
    fragCode += fragInputs_u_matTexRm;
    fragCode += fragInputs_u_matTexMm;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromNm;
    fragCode += fragMainCook_1_MatFromMaterial;
    fragCode += fragMainCook_2_LightLoop;
    if (doEnvMap)
        fragCode += fragMainCook_3_FragColorEm;
    else
        fragCode += fragMainCook_3_FragColor;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCookDmSm(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
in      vec2        v_uv1;      // Texture coordinate varying
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    if (doEnvMap)
        fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexRm;
    fragCode += fragInputs_u_matTexMm;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainCook_1_MatFromDmRmMm;
    fragCode += fragMainCook_2_LightLoopSm;
    if (doEnvMap)
        fragCode += fragMainCook_3_FragColorDmEm;
    else
        fragCode += fragMainCook_3_FragColorTm;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCookDmOm(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertInputs_a_uv1;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
in      vec2        v_uv1;      // Texture coordinate varying
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexRm;
    fragCode += fragInputs_u_matTexMm;
    fragCode += fragInputs_u_matTexOm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainCook_1_MatFromDmRmMm;
    fragCode += fragMainCook_2_LightLoop;
    fragCode += fragMainCook_3_FragColorDmOmEm;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCookDmNm(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
in      vec2        v_uv1;      // Texture coordinate varying
in      vec3        v_eyeDirTS; // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
)";
    fragCode += fragInputs_u_lightAll;
    if (doEnvMap)
        fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexNm;
    fragCode += fragInputs_u_matTexRm;
    fragCode += fragInputs_u_matTexMm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromNm;
    fragCode += fragMainCook_1_MatFromDmRmMm;
    fragCode += fragMainCook_2_LightLoopNm;
    if (doEnvMap)
        fragCode += fragMainCook_3_FragColorDmEm;
    else
        fragCode += fragMainCook_3_FragColorTm;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCookSm(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    fragCode += fragInputs_u_matCookAll;
    if (doEnvMap)
        fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainCook_1_MatFromMaterial;
    fragCode += fragMainCook_2_LightLoopSm;
    if (doEnvMap)
        fragCode += fragMainCook_3_FragColorEm;
    else
        fragCode += fragMainCook_3_FragColor;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCookOm(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertInputs_a_uv1;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
in      vec2        v_uv1;      // Texture coordinate varying
)";
    fragCode += fragInputs_u_lightAll;
    if (doEnvMap)
        fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_matTexRm;
    fragCode += fragInputs_u_matTexMm;
    fragCode += fragInputs_u_matTexOm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainCook_1_MatFromMaterial;
    fragCode += fragMainCook_2_LightLoop;
    if (doEnvMap)
        fragCode += fragMainCook_3_FragColorOmEm;
    else
        fragCode += fragMainCook_3_FragColorOm;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCookDm(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertInputs_a_uv1;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
in      vec2        v_uv1;      // Texture coordinate varying
)";
    fragCode += fragInputs_u_lightAll;
    if (doEnvMap)
        fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexRm;
    fragCode += fragInputs_u_matTexMm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainCook_1_MatFromDmRmMm;
    fragCode += fragMainCook_2_LightLoop;
    if (doEnvMap)
        fragCode += fragMainCook_3_FragColorDmEm;
    else
        fragCode += fragMainCook_3_FragColorTm;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCookNm(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
in      vec2        v_uv1;      // Texture coordinate varying
in      vec3        v_eyeDirTS; // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
)";
    fragCode += fragInputs_u_lightAll;
    if (doEnvMap)
        fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_matCookAll;
    fragCode += fragInputs_u_matTexNm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromNm;
    fragCode += fragMainCook_1_MatFromMaterial;
    fragCode += fragMainCook_2_LightLoop;
    if (doEnvMap)
        fragCode += fragMainCook_3_FragColorEm;
    else
        fragCode += fragMainCook_3_FragColor;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixCook(SLVLight* lights, bool doEnvMap)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_matrix_invMv;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_R_OS;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_R_OS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec3        v_R_OS;     // Interpol. reflect in object space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_matCookAll;
    if (doEnvMap)
        fragCode += fragInputs_u_matCookEnvMaps;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingCookTorrence;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainCook_1_MatFromMaterial;
    fragCode += fragMainCook_2_LightLoop;
    if (doEnvMap)
        fragCode += fragMainCook_3_FragColorEm;
    else
        fragCode += fragMainCook_3_FragColor;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnDmNmOmSm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_uv1;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec2        v_uv1;      // Texture coordinate 1 varying for diffuse color
in      vec2        v_uv2;      // Texture coordinate 2 varying for AO
in      vec3        v_eyeDirTS;                 // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexNm;
    fragCode += fragInputs_u_matTexOm;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromNm;
    fragCode += fragMainBlinn_2_LightLoopNmSm;
    fragCode += fragMainBlinn_3_FragColorTmAo;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnDmNmOm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec2        v_uv1;      // Texture coordinate 1 varying for diffuse color
in      vec2        v_uv2;      // Texture coordinate 2 varying for AO
in      vec3        v_eyeDirTS;                 // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexNm;
    fragCode += fragInputs_u_matTexOm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromNm;
    fragCode += fragMainBlinn_2_LightLoopNm;
    fragCode += fragMainBlinn_3_FragColorTmAo;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnDmNmSm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec2        v_uv1;      // Texture coordinate varying
in      vec3        v_eyeDirTS;                 // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexNm;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromNm;
    fragCode += fragMainBlinn_2_LightLoopNmSm;
    fragCode += fragMainBlinn_3_FragColorTm;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnDmOmSm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec2        v_uv1;      // Interpol. texture coordinate
in      vec2        v_uv2;      // Texture coordinate 2 varying for AO
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexOm;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoopSm;
    fragCode += fragMainBlinn_3_FragColorTmAo;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnDmSm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec2        v_uv1;      // Interpol. texture coordinate
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoopSm;
    fragCode += fragMainBlinn_3_FragColorTm;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnNmSm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec2        v_uv1;      // Texture coordinate varying
in      vec3        v_eyeDirTS;                 // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matTexNm;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromNm;
    fragCode += fragMainBlinn_2_LightLoopNmSm;
    fragCode += fragMainBlinn_3_FragColor;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnOmSm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec2        v_uv2;      // Texture coordinate 2 varying for AO
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matTexOm;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoopSm;
    fragCode += fragMainBlinn_3_FragColorAo;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnNmOm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec2        v_uv1;      // Texture coordinate 1 varying for normal map
in      vec2        v_uv2;      // Texture coordinate 2 varying for AO
in      vec3        v_eyeDirTS;                 // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matTexNm;
    fragCode += fragInputs_u_matTexOm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromNm;
    fragCode += fragMainBlinn_2_LightLoopNm;
    fragCode += fragMainBlinn_3_FragColorAo;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnDmOm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec2        v_uv1;      // Interpol. texture coordinate
in      vec2        v_uv2;      // Texture coordinate 2 varying for AO
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexOm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoop;
    fragCode += fragMainBlinn_3_FragColorTmAo;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnDmNm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;                     // Interpol. point of illumination in view space (VS)
in      vec2        v_uv1;                      // Texture coordinate varying
in      vec3        v_eyeDirTS;                 // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexNm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromNm;
    fragCode += fragMainBlinn_2_LightLoopNm;
    fragCode += fragMainBlinn_3_FragColorTm;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnSm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoopSm;
    fragCode += fragMainBlinn_3_FragColor;
    fragCode += fragMain_4_ColoredShadows;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnOm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
in      vec2        v_uv2;      // Texture coordinate 2 varying for AO
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matTexOm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoop;
    fragCode += fragMainBlinn_3_FragColorAo;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnNm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_TBN_Nm;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;                     // Interpol. point of illumination in view space (VS)
in      vec2        v_uv1;                      // Texture coordinate varying
in      vec3        v_eyeDirTS;                 // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to light 0 in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matTexNm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromNm;
    fragCode += fragMainBlinn_2_LightLoopNm;
    fragCode += fragMainBlinn_3_FragColor;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnDm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_v_uv1;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3   v_P_VS;  // Interpol. point of illumination in view space (VS)
in      vec3   v_N_VS;  // Interpol. normal at v_P_VS in view space
in      vec2   v_uv1;   // Interpol. texture coordinate
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoop;
    fragCode += fragMainBlinn_3_FragColorTm;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinn(SLVLight* lights)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_matBlinnAll;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionsLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMain_Begin;
    fragCode += fragMain_0_Intensities;
    fragCode += fragMain_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoop;
    fragCode += fragMainBlinn_3_FragColor;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixVideoBkgdSm(SLVLight* lights)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_u_matrices_all;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertMain_Begin;
    vertCode += vertMain_v_P_VS;
    vertCode += vertMain_v_P_WS_Sm;
    vertCode += vertMain_v_N_VS;
    vertCode += vertMain_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragInputs_u_matAmbi;
    fragCode += fragInputs_u_matTexDm;
    fragCode += fragInputs_u_matGetsSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragFunctionShadowTest(lights);
    fragCode += fragFunctionDoColoredShadows;
    fragCode += fragMainVideoBkgd;
    fragCode += fragMain_5_FogGammaStereo;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//! Returns true if at least one of the light does shadow mapping
bool SLGLProgramGenerated::lightsDoShadowMapping(SLVLight* lights)
{
    for (auto light : *lights)
    {
        if (light->createsShadows())
            return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
string SLGLProgramGenerated::fragInputs_u_lightSm(SLVLight* lights)
{
    string u_lightSm = R"(
uniform vec4        u_lightPosWS[NUM_LIGHTS];               // position of light in world space
uniform bool        u_lightCreatesShadows[NUM_LIGHTS];      // flag if light creates shadows
uniform int         u_lightNumCascades[NUM_LIGHTS];         // number of cascades for cascaded shadowmap
uniform bool        u_lightDoSmoothShadows[NUM_LIGHTS];     // flag if percentage-closer filtering is enabled
uniform int         u_lightSmoothShadowLevel[NUM_LIGHTS];   // radius of area to sample for PCF
uniform float       u_lightShadowMinBias[NUM_LIGHTS];       // min. shadow bias value at 0° to N
uniform float       u_lightShadowMaxBias[NUM_LIGHTS];       // min. shadow bias value at 90° to N
uniform bool        u_lightUsesCubemap[NUM_LIGHTS];         // flag if light has a cube shadow map
uniform bool        u_lightsDoColoredShadows;               // flag if shadows should be colored
)";
    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLLight* light = lights->at(i);
        if (light->createsShadows())
        {
            SLShadowMap* shadowMap = light->shadowMap();

            if (shadowMap->useCubemap())
            {
                u_lightSm += "uniform mat4        u_lightSpace_" + std::to_string(i) + "[6];\n";
            }
            else if (light->doCascadedShadows())
            {
                u_lightSm += "uniform mat4        u_lightSpace_" + std::to_string(i) + "[" + std::to_string(shadowMap->numCascades()) + "];\n";
            }
            else
            {
                u_lightSm += "uniform mat4        u_lightSpace_" + std::to_string(i) + ";\n";
            }
        }
    }
    return u_lightSm;
}
//-----------------------------------------------------------------------------
string SLGLProgramGenerated::fragInputs_u_shadowMaps(SLVLight* lights)
{
    string smDecl;
    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLLight* light = lights->at(i);
        if (light->createsShadows())
        {
            SLShadowMap* shadowMap = light->shadowMap();
            if (shadowMap->useCubemap())
                smDecl += "uniform samplerCube u_shadowMapCube_" + to_string(i) + ";\n";
            else if (light->doCascadedShadows())
            {
                for (int j = 0; j < light->shadowMap()->depthBuffers().size(); j++)
                    smDecl += "uniform sampler2D   u_cascadedShadowMap_" + to_string(i) + "_" + std::to_string(j) + ";\n";

                smDecl += "uniform float       u_cascadesFactor_" + to_string(i) + ";\n";
            }
            else
                smDecl += "uniform sampler2D   u_shadowMap_" + to_string(i) + ";\n";
        }
    }
    return smDecl;
}
//-----------------------------------------------------------------------------
//! Adds the core shadow mapping test routine depending on the lights
string SLGLProgramGenerated::fragFunctionShadowTest(SLVLight* lights)
{
    bool doCascadedSM = false;
    for (SLLight* light : *lights)
    {
        if (light->doCascadedShadows())
        {
            doCascadedSM = true;
            break;
        }
    }

    string shadowTestCode = R"(
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

int getCascadesDepthIndex(in int i, int numCascades)
{
    float factor;
)";

    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLShadowMap* shadowMap = lights->at(i)->shadowMap();
        if (shadowMap && shadowMap->useCascaded())
        {
            shadowTestCode += "    if (i == " + std::to_string(i) + ")\n";
            shadowTestCode += "    {\n";
            shadowTestCode += "        factor = u_cascadesFactor_" + std::to_string(i) + ";\n";
            shadowTestCode += "    }\n";
        }
    }

    shadowTestCode += R"(
    float fi = u_camClipNear;
    float ni;

    for (int i = 0; i < numCascades-1; i++)
    {
        ni = fi;
        fi = factor * u_camClipNear * pow((u_camClipFar/(factor*u_camClipNear)), float(i+1)/float(numCascades));
        if (-v_P_VS.z < fi)
            return i;
    }
    return numCascades-1;
}

float shadowTest(in int i, in vec3 N, in vec3 lightDir)
{
    if (u_lightCreatesShadows[i])
    {
        // Calculate position in light space
        mat4 lightSpace;
        vec3 lightToFragment = v_P_WS - u_lightPosWS[i].xyz;
)";

    if (doCascadedSM > 0)
    {
        shadowTestCode += R"(
        int index = 0;

        if (u_lightNumCascades[i] > 0)
        {
            index = getCascadesDepthIndex(i, u_lightNumCascades[i]);
        )";
        for (SLuint i = 0; i < lights->size(); ++i)
        {
            SLShadowMap* shadowMap = lights->at(i)->shadowMap();
            if (shadowMap && shadowMap->useCascaded())
                shadowTestCode += "    if (i == " + std::to_string(i) + ") { lightSpace = u_lightSpace_" + std::to_string(i) + "[index]; }\n";
        }
        shadowTestCode += R"(
        }
        else if (u_lightUsesCubemap[i])
        {)";
        for (SLuint i = 0; i < lights->size(); ++i)
        {
            SLShadowMap* shadowMap = lights->at(i)->shadowMap();
            if (shadowMap && shadowMap->useCubemap())
                shadowTestCode += "    if (i == " + std::to_string(i) + ") { lightSpace = u_lightSpace_" + std::to_string(i) + "[vectorToFace(lightToFragment)]; }\n";
        }
        shadowTestCode += R"(
        }
        else
        {
        )";
        for (SLuint i = 0; i < lights->size(); ++i)
        {
            SLShadowMap* shadowMap = lights->at(i)->shadowMap();
            if (shadowMap && !shadowMap->useCubemap() && !shadowMap->useCascaded())
                shadowTestCode += "if (i == " + std::to_string(i) + ") { lightSpace = u_lightSpace_" + std::to_string(i);
            +"}\n";
        }
        shadowTestCode += R"(
        }
        )";
    }
    else
    {
        shadowTestCode += R"(
        if (u_lightUsesCubemap[i])
        {
)";
        for (SLuint i = 0; i < lights->size(); ++i)
        {
            SLShadowMap* shadowMap = lights->at(i)->shadowMap();
            if (shadowMap && shadowMap->useCubemap())
                shadowTestCode += "            if (i == " + std::to_string(i) + ") lightSpace = u_lightSpace_" + std::to_string(i) + "[vectorToFace(lightToFragment)];\n";
        }
        shadowTestCode += R"(
        }
        else
        {
)";
        for (SLuint i = 0; i < lights->size(); ++i)
        {
            SLShadowMap* shadowMap = lights->at(i)->shadowMap();
            if (shadowMap && !shadowMap->useCubemap() && !shadowMap->useCascaded())
                shadowTestCode += "            if (i == " + std::to_string(i) + ") lightSpace = u_lightSpace_" + std::to_string(i) + ";\n";
        }
        shadowTestCode += R"(
        }
        )";
    }

    shadowTestCode += R"(
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
        if (u_lightDoSmoothShadows[i])
        {
            int level = u_lightSmoothShadowLevel[i];
            vec2 texelSize;
)";

    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLShadowMap* shadowMap = lights->at(i)->shadowMap();
        if (!shadowMap->useCascaded() && !shadowMap->useCubemap())
            shadowTestCode += "            if (i == " + to_string(i) + ") { texelSize = 1.0 / vec2(textureSize(u_shadowMap_" + to_string(i) + ", 0)); }\n";
        else if (shadowMap->useCascaded())
        {
            shadowTestCode += "            if (i == " + to_string(i) + ")\n            {\n";
            for (int j = 0; j < shadowMap->depthBuffers().size(); j++)
                shadowTestCode += "                if (index == " + to_string(j) + ") { texelSize = 1.0 / vec2(textureSize(u_cascadedShadowMap_" + to_string(i) + "_" + to_string(j) + ", 0)); }\n";

            shadowTestCode += "            }\n";
        }
    }
    shadowTestCode += R"(
            for (int x = -level; x <= level; ++x)
            {
                for (int y = -level; y <= level; ++y)
                {
    )";
    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLShadowMap* shadowMap = lights->at(i)->shadowMap();
        if (!shadowMap->useCascaded() && !shadowMap->useCubemap())
            shadowTestCode += "                 if (i == " + to_string(i) + ") { closestDepth = texture(u_shadowMap_" + to_string(i) + ", projCoords.xy + vec2(x, y) * texelSize).r; }\n";
        else if (shadowMap->useCascaded())
        {
            shadowTestCode += "                if (i == " + to_string(i) + ")\n                    {\n";
            for (int j = 0; j < shadowMap->depthBuffers().size(); j++)
                shadowTestCode += "                        if (index == " + to_string(j) + ") { closestDepth = texture(u_cascadedShadowMap_" + to_string(i) + "_" + to_string(j) + ", projCoords.xy + vec2(x, y) * texelSize).r; }\n";
            shadowTestCode += "                    }\n";
        }
    }

    shadowTestCode += R"(
                    shadow += currentDepth - bias > closestDepth ? 1.0 : 0.0;
                }
            }
            shadow /= pow(1.0 + 2.0 * float(level), 2.0);
        }
        else
        {
            if (u_lightUsesCubemap[i])
            {
)";
    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLShadowMap* shadowMap = lights->at(i)->shadowMap();
        if (shadowMap->useCubemap())
            shadowTestCode += "                if (i == " + to_string(i) + ") closestDepth = texture(u_shadowMapCube_" + to_string(i) + ", lightToFragment).r;\n";
    }
    shadowTestCode += R"(
            }
            else if (u_lightNumCascades[i] > 0)
            {
)";
    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLLight*     light     = lights->at(i);
        SLShadowMap* shadowMap = lights->at(i)->shadowMap();
        if (light->doCascadedShadows())
        {
            shadowTestCode += "                if (i == " + to_string(i) + ")\n                {\n";
            for (int j = 0; j < shadowMap->depthBuffers().size(); j++)
                shadowTestCode += "                    if (index == " + to_string(j) + ") { closestDepth = texture(u_cascadedShadowMap_" + to_string(i) + "_" + to_string(j) + ", projCoords.xy).r; }\n";
            shadowTestCode += "                }";
        }
    }

    shadowTestCode += R"(
            }
            else
            {
)";

    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLShadowMap* shadowMap = lights->at(i)->shadowMap();
        if (!shadowMap->useCubemap() && !shadowMap->useCascaded())
            shadowTestCode += "                if (i == " + to_string(i) + ") closestDepth = texture(u_shadowMap_" + to_string(i) + ", projCoords.xy).r;\n";
    }

    shadowTestCode += R"(
            }

            // The fragment is in shadow if the light doesn't "see" it
            if (currentDepth > closestDepth + bias)
                shadow = 1.0;
        }

        return shadow;
    }

    return 0.0;
}
)";

    return shadowTestCode;
}
//-----------------------------------------------------------------------------
//! Add vertex shader code to the SLGLShader instance
void SLGLProgramGenerated::addCodeToShader(SLGLShader*   shader,
                                           const string& code,
                                           const string& name)
{

#if defined(DEBUG) && defined(_DEBUG)
    shader->code(SLGLShader::removeComments(code));
#else
    shader->code(code);
#endif
    shader->name(name);

    // Check if generatedShaderPath folder exists
    generatedShaderPath = SLGLProgramManager::configPath + "generatedShaders/";
    if (!Utils::dirExists(SLGLProgramManager::configPath))
        SL_EXIT_MSG("SLGLProgramGenerated::addCodeToShader: SLGLProgramManager::configPath not existing");

    if (!Utils::dirExists(generatedShaderPath))
    {
        bool dirCreated = Utils::makeDir(generatedShaderPath);
        if (!dirCreated)
            SL_EXIT_MSG("SLGLProgramGenerated::addCodeToShader: Failed to created SLGLProgramManager::configPath/generatedShaders");
    }

    shader->file(generatedShaderPath + name);
}
//-----------------------------------------------------------------------------
//! Adds shader header code
string SLGLProgramGenerated::shaderHeader(int numLights)

{
    string header = "\nprecision highp float;\n";
    header += "\n#define NUM_LIGHTS " + to_string(numLights) + "\n";
    return header;
}
//-----------------------------------------------------------------------------
