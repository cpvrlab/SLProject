//#############################################################################
//  File:      SLGLProgramGenerated.cpp
//  Author:    Marcus Hudritsch
//  Date:      December 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//             This software is provide under the GNU General Public License
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
const string vertInputs_a_pn = R"(
layout (location = 0) in vec4  a_position;  // Vertex position attribute
layout (location = 1) in vec3  a_normal;    // Vertex normal attribute)";
const string vertInputs_a_uv1 = R"(
layout (location = 2) in vec2  a_uv1;       // Vertex tex.coord. 1 for diffuse color)";
const string vertInputs_a_uv2 = R"(
layout (location = 3) in vec2  a_uv2;       // Vertex tex.coord. 2 for AO)";
const string vertInputs_a_tangent = R"(
layout (location = 5) in vec4  a_tangent;   // Vertex tangent attribute
)";
//-----------------------------------------------------------------------------
const string vertInputs_u_matrices = R"(
uniform mat3  u_nMatrix;    // normal matrix=transpose(inverse(mv))
uniform mat4  u_mMatrix;    // model matrix
uniform mat4  u_mvMatrix;   // modelview matrix
uniform mat4  u_mvpMatrix;  // = projection * modelView
)";
//-----------------------------------------------------------------------------
const string vertInputs_u_lightNm = R"(
uniform vec4  u_lightPosVS[NUM_LIGHTS];     // position of light in view space
uniform vec3  u_lightSpotDir[NUM_LIGHTS];   // spot direction in view space
uniform float u_lightSpotDeg[NUM_LIGHTS];   // spot cutoff angle 1-180 degrees
)";
//-----------------------------------------------------------------------------
const string vertOutputs_v_P_VS = R"(
out     vec3  v_P_VS;                   // Point of illumination in view space (VS))";
const string vertOutputs_v_N_VS = R"(
out     vec3  v_N_VS;                   // Normal at P_VS in view space (VS))";
const string vertOutputs_v_P_WS = R"(
out     vec3  v_P_WS;                   // Point of illumination in world space (WS))";
const string vertOutputs_v_uv1 = R"(
out     vec2  v_uv1;                    // Texture coordinate 1 output)";
const string vertOutputs_v_uv2 = R"(
out     vec2  v_uv2;                    // Texture coordinate 1 output)";
const string vertOutputs_v_lightNm = R"(
out     vec3  v_eyeDirTS;               // Vector to the eye in tangent space
out     vec3  v_lightDirTS[NUM_LIGHTS]; // Vector to the light 0 in tangent space
out     vec3  v_spotDirTS[NUM_LIGHTS];  // Spot direction in tangent space
)";
//-----------------------------------------------------------------------------
const string vertMainBlinn_BeginAll = R"(
void main()
{
    v_P_VS = vec3(u_mvMatrix *  a_position); // vertex position in view space)";
const string vertMainBlinn_v_P_WS_Sm = R"(
    v_P_WS = vec3(u_mMatrix * a_position);   // vertex position in world space)";
const string vertMainBlinn_v_N_VS = R"(
    v_N_VS = vec3(u_nMatrix * a_normal);     // vertex normal in view space)";
const string vertMainBlinn_v_uv1 = R"(
    v_uv1 = a_uv1;  // pass diffuse color tex.coord. 1 for interpolation)";
const string vertMainBlinn_v_uv2_Ao = R"(
    v_uv2 = a_uv2;  // pass ambient occlusion tex.coord. 2 for interpolation)";
const string vertMainBlinn_TBN_Nm = R"(
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
const string vertMainBlinn_EndAll = R"(
    // pass the vertex w. the fix-function transform
    gl_Position = u_mvpMatrix * a_position;
}
)";
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
const string fragInputs_u_lightAll = R"(
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
//-----------------------------------------------------------------------------



const string fragInputs_u_lightSc = R"(
)";

//-----------------------------------------------------------------------------
const string fragInputs_u_matAllBlinn = R"(
uniform vec4        u_matAmbi;          // ambient color reflection coefficient (ka)
uniform vec4        u_matDiff;          // diffuse color reflection coefficient (kd)
uniform vec4        u_matSpec;          // specular color reflection coefficient (ks)
uniform vec4        u_matEmis;          // emissive color for self-shining materials
uniform float       u_matShin;          // shininess exponent
)";
//-----------------------------------------------------------------------------
const string fragInputs_u_matTm = R"(
uniform sampler2D   u_matTexture0;      // diffuse color map
)";
const string fragInputs_u_matAo = R"(
uniform sampler2D   u_matTexture0;      // ambient occlusion map
)";
const string fragInputs_u_matSm = R"(
uniform bool        u_matGetsShadows;   // flag if material receives shadows
)";
const string fragInputs_u_matTmNm = R"(
uniform sampler2D   u_matTexture0;      // diffuse color map
uniform sampler2D   u_matTexture1;      // normal bump map
)";
const string fragInputs_u_matTmPm = R"(
uniform sampler2D   u_matTexture0;      // diffuse color map
uniform sampler2D   u_matTexture1;      // normal bump map
uniform sampler2D   u_matTexture2;      // normal bump map
)";
const string fragInputs_u_matTmAo = R"(
uniform sampler2D   u_matTexture0;      // diffuse color map
uniform sampler2D   u_matTexture1;      // ambient occlusion map
)";
const string fragInputs_u_matTmSm = R"(
uniform sampler2D   u_matTexture0;      // diffuse color map
uniform bool        u_matGetsShadows;   // flag if material receives shadows
)";
const string fragInputs_u_matNmSm = R"(
uniform sampler2D   u_matTexture0;      // normal bump map
uniform bool        u_matGetsShadows;   // flag if material receives shadows
)";
const string fragInputs_u_matAoSm = R"(
uniform sampler2D   u_matTexture0;      // ambient occlusion map
uniform bool        u_matGetsShadows;   // flag if material receives shadows
)";
const string fragInputs_u_matNmAo = R"(
uniform sampler2D   u_matTexture0;      // normal bump map
uniform sampler2D   u_matTexture1;      // ambient occlusion map
)";
const string fragInputs_u_matTmNmAo = R"(
uniform sampler2D   u_matTexture0;      // diffuse color map
uniform sampler2D   u_matTexture1;      // normal bump map
uniform sampler2D   u_matTexture2;      // ambient occlusion map
)";
const string fragInputs_u_matTmNmSm = R"(
uniform sampler2D   u_matTexture0;      // diffuse color map
uniform sampler2D   u_matTexture1;      // normal bump map
uniform bool        u_matGetsShadows;   // flag if material receives shadows
)";
const string fragInputs_u_matTmAoSm = R"(
uniform sampler2D   u_matTexture0;      // diffuse color map
uniform sampler2D   u_matTexture1;      // ambient occlusion map
uniform bool        u_matGetsShadows;   // flag if material receives shadows
)";
const string fragInputs_u_matTmNmAoSm = R"(
uniform sampler2D   u_matTexture0;      // diffuse color map
uniform sampler2D   u_matTexture1;      // normal bump map
uniform sampler2D   u_matTexture2;      // ambient occlusion map
uniform bool        u_matGetsShadows;   // flag if material receives shadows
)";
//-----------------------------------------------------------------------------
const string fragInputs_u_cam = R"(
uniform float       u_camNearPlane;     // near plane distance
uniform float       u_camFarPlane;      // far plane distance
uniform int         u_camProjection;    // type of stereo
uniform int         u_camStereoEye;     // -1=left, 0=center, 1=right
uniform mat3        u_camStereoColors;  // color filter matrix
uniform bool        u_camFogIsOn;       // flag if fog is on
uniform int         u_camFogMode;       // 0=LINEAR, 1=EXP, 2=EXP2
uniform float       u_camFogDensity;    // fog density value
uniform float       u_camFogStart;      // fog start distance
uniform float       u_camFogEnd;        // fog end distance
uniform vec4        u_camFogColor;      // fog color (usually the background)
)";

//-----------------------------------------------------------------------------
const string fragInputs_u_camForCascaded = R"(
    uniform float u_camCascadeDepth[NUM_CASCADES];
)";

//-----------------------------------------------------------------------------
const string fragOutputs_o_fragColor = R"(
out     vec4        o_fragColor;        // output fragment color
)";

//-----------------------------------------------------------------------------
const string fragFunctionColoredCascadedShadow = R"(
void coloredCascadedShadow(in int i, in float shadow, inout vec4 Id, inout vec4 Is)


)";
//-----------------------------------------------------------------------------
const string fragFunctionLightingBlinnPhong = R"(
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

    // accumulate directional light intesities w/o attenuation
    Ia += u_lightAmbi[i];
    Id += u_lightDiff[i]  * diffFactor * (1.0 - shadow);
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
const string fragFunctionLightingCookTorrance = R"(
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

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

float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float geometrySmith(vec3 N, vec3 E, vec3 L, float roughness)
{
    float NdotV = max(dot(N, E), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = geometrySchlickGGX(NdotV, roughness);
    float ggx1  = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

void directLightCookTorrance(in    int   i,        // Light index
                             in    vec3  N,        // Normalized normal at v_P_VS
                             in    vec3  E,        // Normalized vector from v_P to the eye
                             in    vec3  S,        // Normalized light spot direction
                             in    vec3  lightDiff,// diffuse light intensity
                             in    vec3  matDiff,  // diffuse material reflection
                             in    float matMetal, // diffuse material reflection
                             in    float matRough, // diffuse material reflection
                             inout vec3  Lo)       // reflected intensity
{
    vec3 H = normalize(E + S);  // Normalized halfvector between eye and light vector

    vec3 radiance = lightDiff;  // Per light radiance without attenuation

    // Init Fresnel reflection at 90 deg. (0 to N)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, matDiff, matMetal);

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

    Lo += (kD*matDiff.rgb/PI + specular) * radiance * NdotL;
}

void pointLightCookTorrance(in    int   i,        // Light index
                            in    vec3  N,        // Normalized normal at v_P_VS
                            in    vec3  E,        // Normalized vector from v_P to the eye
                            in    vec3  L,        // Vector from v_P to the light
                            in    vec3  S,        // Normalized light spot direction
                            in    vec3  lightDiff,// diffuse light intensity
                            in    vec3  matDiff,  // diffuse material reflection
                            in    float matMetal, // diffuse material reflection
                            in    float matRough, // diffuse material reflection
                            inout vec3  Lo)       // reflected intensity
{
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

    vec3 radiance = lightDiff * att;        // per light radiance

    // Init Fresnel reflection at 90 deg. (0 to N)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, matDiff, matMetal);

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

    Lo += (kD*matDiff.rgb/PI + specular) * radiance * NdotL;
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
const string fragMainBlinn_0_IntensityDeclaration = R"(
void main()
{
    vec4 Ia = vec4(0.0); // Accumulated ambient light intensity at v_P_VS
    vec4 Id = vec4(0.0); // Accumulated diffuse light intensity at v_P_VS
    vec4 Is = vec4(0.0); // Accumulated specular light intensity at v_P_VS
)";
//-----------------------------------------------------------------------------
const string fragMainBlinn_1_EN_fromVert = R"(
    vec3 E = normalize(-v_P_VS); // Interpolated vector from p to the eye
    vec3 N = normalize(v_N_VS);  // A input normal has not anymore unit length
)";
const string fragMainBlinn_1_EN_fromNm0 = R"(
    vec3 E = normalize(v_eyeDirTS);   // normalized interpolated eye direction
    // Get normal from normal map, move from [0,1] to [-1, 1] range & normalize
    vec3 N = normalize(texture(u_matTexture0, v_uv1).rgb * 2.0 - 1.0);
)";
const string fragMainBlinn_1_EN_fromNm1 = R"(
    vec3 E = normalize(v_eyeDirTS);   // normalized interpolated eye direction
    // Get normal from normal map, move from [0,1] to [-1, 1] range & normalize
    vec3 N = normalize(texture(u_matTexture1, v_uv1).rgb * 2.0 - 1.0);
)";
const string fragMainBlinn_1_EN_fromNm1Hm2 = R"(
    vec3 E = normalize(v_eyeDirTS);   // normalized interpolated eye direction

    // Calculate new texture coord. Tc for Parallax mapping
    // The height comes from red channel from the height map
    float height = texture(u_matTexture2, v_uv1.st).r;

    // Scale the height and add the bias (height offset)
    height = height * u_scale + u_offset;

    // Add the texture offset to the texture coord.
    vec2 Tc = v_uv1.st + (height * E.st);

    // Get normal from normal map, move from [0,1] to [-1, 1] range & normalize
    vec3 N = normalize(texture(u_matTexture1, Tc).rgb * 2.0 - 1.0);
)";

const string indexToColor = R"(
    vec3 indexToColor(int index)
    {
        if (index == 0)      { return vec3(1.0, 0.0, 0.0); }
        else if (index == 1) { return vec3(0.0, 1.0, 0.0); }
        else if (index == 2) { return vec3(0.0, 0.0, 1.0); }
        else if (index == 3) { return vec3(1.0, 1.0, 0.0); }
        else if (index == 4) { return vec3(0.0, 1.0, 1.0); }
        else if (index == 5) { return vec3(1.0, 0.0, 1.0); }
    }
)";

//-----------------------------------------------------------------------------
const string fragMainBlinn_2_LightLoop = R"(
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
const string fragMainBlinn_2_LightLoopNm = R"(
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

const string fragMainBlinn_2_LightLoopSm = R"(
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
//-----------------------------------------------------------------------------
const string fragMainBlinn_3_FragColor = R"(
    // Sum up all the reflected color components
    o_fragColor =  u_matEmis +
                   u_globalAmbi +
                   Ia * u_matAmbi +
                   Id * u_matDiff +
                   Is * u_matSpec;

    // For correct alpha blending overwrite alpha component
    o_fragColor.a = u_matDiff.a;
)";
const string fragMainBlinn_3_FragColorTm = R"(
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
)";
const string fragMainBlinn_3_FragColorAo0 = R"(
    // Get ambient occlusion factor
    float AO = texture(u_matTexture0, v_uv2).r;

    // Sum up all the reflected color components
    o_fragColor =  u_matEmis +
                   u_globalAmbi +
                   Ia * u_matAmbi * AO +
                   Id * u_matDiff +
                   Is * u_matSpec;

    // For correct alpha blending overwrite alpha component
    o_fragColor.a = u_matDiff.a;
)";
const string fragMainBlinn_3_FragColorAo1 = R"(
    // Get ambient occlusion factor
    float AO = texture(u_matTexture1, v_uv2).r;

    // Sum up all the reflected color components
    o_fragColor =  u_matEmis +
                   u_globalAmbi +
                   Ia * u_matAmbi * AO +
                   Id * u_matDiff +
                   Is * u_matSpec;

    // For correct alpha blending overwrite alpha component
    o_fragColor.a = u_matDiff.a;
)";
const string fragMainBlinn_3_FragColorAo1Tm = R"(
    // Get ambient occlusion factor
    float AO = texture(u_matTexture1, v_uv2).r;

    // Sum up all the reflected color components
    o_fragColor =  u_matEmis +
                   u_globalAmbi +
                   Ia * u_matAmbi * AO +
                   Id * u_matDiff;

    // Componentwise multiply w. texture color
    o_fragColor *= texture(u_matTexture0, v_uv1);

    // add finally the specular RGB-part
    vec4 specColor = Is * u_matSpec;
    o_fragColor.rgb += specColor.rgb;
)";
const string fragMainBlinn_3_FragColorAo2Tm = R"(
    // Get ambient occlusion factor
    float AO = texture(u_matTexture2, v_uv2).r;

    // Sum up all the reflected color components
    o_fragColor =  u_matEmis +
                   u_globalAmbi +
                   Ia * u_matAmbi * AO +
                   Id * u_matDiff;

    // Componentwise multiply w. texture color
    o_fragColor *= texture(u_matTexture0, v_uv1);

    // add finally the specular RGB-part
    vec4 specColor = Is * u_matSpec;
    o_fragColor.rgb += specColor.rgb;
)";
//-----------------------------------------------------------------------------
const string fragMainBlinn_4_End = R"(
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

    bool matHasTm = mat->hasTextureType(TT_diffuse);
    bool matHasNm = mat->hasTextureType(TT_normal);
    bool matHasHm = mat->hasTextureType(TT_height);
    bool matHasAo = mat->hasTextureType(TT_ambientOcclusion);

    if (mat->lightModel() == LM_BlinnPhong)
        programName += "PerPixBlinn";
    else if (mat->lightModel() == LM_CookTorrance)
        programName += "PerPixCook";
    else
        programName += "Custom";
    if (matHasTm)
        programName += "Tm";
    //if (matHasNm && matHasHm)
    //    programName += "Pm";
    if (matHasNm && !matHasHm)
        programName += "Nm";
    if (matHasAo)
        programName += "Ao";
    programName += "-";

    // Add letter per light type
    for (auto light : *lights)
    {
        if (light->positionWS().w == 0.0f)
        {
            if (light->doCascadedShadows())
                programName += "C"; // Directional light with cascaded shadowmap
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
    assert(mat && "No material pointer passed!");
    assert(!lights->empty() && "No lights passed!");
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    // Check what textures the material has
    bool Tm = mat->hasTextureType(TT_diffuse);
    bool Nm = mat->hasTextureType(TT_normal);
    bool Hm = mat->hasTextureType(TT_height);
    bool Ao = mat->hasTextureType(TT_ambientOcclusion);

    // Check if any of the scene lights does shadow mapping
    bool Sm = lightsDoShadowMapping(lights);

    if (mat->lightModel() == LM_BlinnPhong)
    {
        if (Tm && Nm && Ao && Sm)
            buildPerPixBlinnTmNmAoSm(lights);
        else if (Tm && Nm && Ao)
            buildPerPixBlinnTmNmAo(lights);
        else if (Tm && Nm && Sm)
            buildPerPixBlinnTmNmSm(lights);
        else if (Tm && Ao && Sm)
            buildPerPixBlinnTmAoSm(lights);
        else if (Ao && Sm)
            buildPerPixBlinnAoSm(lights);
        else if (Nm && Sm)
            buildPerPixBlinnNmSm(lights);
        else if (Tm && Sm)
            buildPerPixBlinnTmSm(lights);
        else if (Nm && Ao)
            buildPerPixBlinnNmAo(lights);
        else if (Tm && Ao)
            buildPerPixBlinnTmAo(lights);
        else if (Tm && Nm)
            buildPerPixBlinnTmNm(lights);
        else if (Sm)
            buildPerPixBlinnSm(lights);
        else if (Ao)
            buildPerPixBlinnAo(lights);
        //else if (Nm && Hm)
        //    buildPerPixBlinnPm(lights);
        else if (Nm)
            buildPerPixBlinnNm(lights);
        else if (Tm)
            buildPerPixBlinnTm(lights);
        else
            buildPerPixBlinn(lights);
    }
    else
        SL_EXIT_MSG("Only Blinn-Phong supported yet.");
}



//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTmNmAoSm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_P_WS_Sm;
    vertCode += vertMainBlinn_v_uv1;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMainBlinn_TBN_Nm;
    vertCode += vertMainBlinn_EndAll;
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
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matTmNmAoSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragShadowTest(lights);
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromNm1;
    fragCode += fragMainBlinn_2_LightLoopNmSm;
    fragCode += fragMainBlinn_3_FragColorAo2Tm;
    fragCode += fragMainBlinn_4_End;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTmNmAo(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertInputs_u_matrices;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_uv1;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMainBlinn_TBN_Nm;
    vertCode += vertMainBlinn_EndAll;
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
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matTmNmAo;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromNm1;
    fragCode += fragMainBlinn_2_LightLoopNm;
    fragCode += fragMainBlinn_3_FragColorAo2Tm;
    fragCode += fragMainBlinn_4_End;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTmNmSm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_P_WS_Sm;
    vertCode += vertMainBlinn_v_uv1;
    vertCode += vertMainBlinn_TBN_Nm;
    vertCode += vertMainBlinn_EndAll;
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
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matTmNmSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragShadowTest(lights);
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromNm1;
    fragCode += fragMainBlinn_2_LightLoopNmSm;
    fragCode += fragMainBlinn_3_FragColorTm;
    fragCode += fragMainBlinn_4_End;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTmAoSm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_u_matrices;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_P_WS_Sm;
    vertCode += vertMainBlinn_v_N_VS;
    vertCode += vertMainBlinn_v_uv1;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMainBlinn_EndAll;
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
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matTmAoSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragShadowTest(lights);
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoopSm;
    fragCode += fragMainBlinn_3_FragColorAo1Tm;
    fragCode += fragMainBlinn_4_End;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTmSm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_u_matrices;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_P_WS_Sm;
    vertCode += vertMainBlinn_v_N_VS;
    vertCode += vertMainBlinn_v_uv1;
    vertCode += vertMainBlinn_EndAll;
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
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matTmSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragShadowTest(lights);
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoopSm;
    fragCode += fragMainBlinn_3_FragColorTm;
    fragCode += fragMainBlinn_4_End;
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
    vertCode += vertInputs_u_matrices;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_P_WS_Sm;
    vertCode += vertMainBlinn_v_uv1;
    vertCode += vertMainBlinn_TBN_Nm;
    vertCode += vertMainBlinn_EndAll;
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
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matNmSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragShadowTest(lights);
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromNm0;
    fragCode += fragMainBlinn_2_LightLoopNmSm;
    fragCode += fragMainBlinn_3_FragColor;
    fragCode += fragMainBlinn_4_End;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnAoSm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_u_matrices;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_P_WS_Sm;
    vertCode += vertMainBlinn_v_N_VS;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMainBlinn_EndAll;
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
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matAoSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragShadowTest(lights);
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoopSm;
    fragCode += fragMainBlinn_3_FragColorAo0;
    fragCode += fragMainBlinn_4_End;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnNmAo(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_uv1;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMainBlinn_TBN_Nm;
    vertCode += vertMainBlinn_EndAll;
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
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matNmAo;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromNm0;
    fragCode += fragMainBlinn_2_LightLoopNm;
    fragCode += fragMainBlinn_3_FragColorAo1;
    fragCode += fragMainBlinn_4_End;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTmAo(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_u_matrices;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_N_VS;
    vertCode += vertMainBlinn_v_uv1;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMainBlinn_EndAll;
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
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matTmAo;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoop;
    fragCode += fragMainBlinn_3_FragColorAo1Tm;
    fragCode += fragMainBlinn_4_End;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTmNm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_a_tangent;
    vertCode += vertInputs_u_matrices;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_uv1;
    vertCode += vertMainBlinn_TBN_Nm;
    vertCode += vertMainBlinn_EndAll;
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
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matTmNm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromNm1;
    fragCode += fragMainBlinn_2_LightLoopNm;
    fragCode += fragMainBlinn_3_FragColorTm;
    fragCode += fragMainBlinn_4_End;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnSm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_u_matrices;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_P_WS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_P_WS_Sm;
    vertCode += vertMainBlinn_v_N_VS;
    vertCode += vertMainBlinn_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    int nbCascade = 0;
    for (SLLight * light : *lights)
    {
        if (light->doCascadedShadows())
        {
            int n = light->shadowMap()->depthBuffers().size();
            if (nbCascade != 0 && n != nbCascade)
                std::cout << "error not same number of cascades per light" << std::endl;
            nbCascade = n;
        }
    }

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size(), nbCascade);
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_lightSm(lights);
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matSm;
    fragCode += fragInputs_u_shadowMaps(lights);
    fragCode += fragInputs_u_cam;
    fragCode += indexToColor;
    if (nbCascade > 0)
        fragCode += fragInputs_u_camForCascaded;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragShadowTest(lights);
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoopSm;
    fragCode += fragMainBlinn_3_FragColor;
    //fragCode += coloredShadows(); // enable this to see the different cascades with different colors
    fragCode += fragMainBlinn_4_End;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnAo(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv2;
    vertCode += vertInputs_u_matrices;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv2;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_N_VS;
    vertCode += vertMainBlinn_v_uv2_Ao;
    vertCode += vertMainBlinn_EndAll;
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
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matAo;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoop;
    fragCode += fragMainBlinn_3_FragColorAo0;
    fragCode += fragMainBlinn_4_End;
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
    vertCode += vertInputs_u_matrices;
    vertCode += vertInputs_u_lightNm;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertOutputs_v_lightNm;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_uv1;
    vertCode += vertMainBlinn_TBN_Nm;
    vertCode += vertMainBlinn_EndAll;
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
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matTmNm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromNm1;
    fragCode += fragMainBlinn_2_LightLoopNm;
    fragCode += fragMainBlinn_3_FragColor;
    fragCode += fragMainBlinn_4_End;
    addCodeToShader(_shaders[1], fragCode, _name + ".frag");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTm(SLVLight* lights)
{
    // Assemble vertex shader code
    string vertCode;
    vertCode += shaderHeader((int)lights->size());
    vertCode += vertInputs_a_pn;
    vertCode += vertInputs_a_uv1;
    vertCode += vertInputs_u_matrices;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertOutputs_v_uv1;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_N_VS;
    vertCode += vertMainBlinn_v_uv1;
    vertCode += vertMainBlinn_EndAll;
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
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_matTm;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoop;
    fragCode += fragMainBlinn_3_FragColorTm;
    fragCode += fragMainBlinn_4_End;
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
    vertCode += vertInputs_u_matrices;
    vertCode += vertOutputs_v_P_VS;
    vertCode += vertOutputs_v_N_VS;
    vertCode += vertMainBlinn_BeginAll;
    vertCode += vertMainBlinn_v_N_VS;
    vertCode += vertMainBlinn_EndAll;
    addCodeToShader(_shaders[0], vertCode, _name + ".vert");

    // Assemble fragment shader code
    string fragCode;
    fragCode += shaderHeader((int)lights->size());
    fragCode += R"(
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space
)";
    fragCode += fragInputs_u_lightAll;
    fragCode += fragInputs_u_matAllBlinn;
    fragCode += fragInputs_u_cam;
    fragCode += fragOutputs_o_fragColor;
    fragCode += fragFunctionLightingBlinnPhong;
    fragCode += fragFunctionFogBlend;
    fragCode += fragFunctionDoStereoSeparation;
    fragCode += fragMainBlinn_0_IntensityDeclaration;
    fragCode += fragMainBlinn_1_EN_fromVert;
    fragCode += fragMainBlinn_2_LightLoop;
    fragCode += fragMainBlinn_3_FragColor;
    fragCode += fragMainBlinn_4_End;
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
uniform int         u_lightNbCascades[NUM_LIGHTS];          // number of cascades for cascaded shadowmap
uniform bool        u_lightDoSmoothShadows[NUM_LIGHTS];     // flag if percentage-closer filtering is enabled
uniform int         u_lightSmoothShadowLevel[NUM_LIGHTS];   // radius of area to sample for PCF
uniform float       u_lightShadowMinBias[NUM_LIGHTS];       // min. shadow bias value at 0° to N
uniform float       u_lightShadowMaxBias[NUM_LIGHTS];       // min. shadow bias value at 90° to N
uniform bool        u_lightUsesCubemap[NUM_LIGHTS];         // flag if light has a cube shadow map
)";
    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLLight* light = lights->at(i);
        if (light->createsShadows())
        {
            SLShadowMap* shadowMap = light->shadowMap();

            if (shadowMap->useCubemap())
            {
                u_lightSm += "uniform mat4        u_lightSpace_" + std::to_string(i) + "[6];";
            }
            else if (light->doCascadedShadows())
            {
                u_lightSm += "uniform mat4        u_lightSpace_" + std::to_string(i) + "[" + std::to_string(shadowMap->nbCascades()) + "];";
            }
            else
            {
                u_lightSm += "uniform mat4        u_lightSpace_" + std::to_string(i) + ";";
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
                smDecl += "uniform sampler2D   u_cascadedShadowMap_" + to_string(i) + "[" + std::to_string(light->shadowMap()->depthBuffers().size()) + "];\n";
            else
                smDecl += "uniform sampler2D   u_shadowMap_" + to_string(i) + ";\n";;
        }
    }
    return smDecl;
}
//-----------------------------------------------------------------------------
string SLGLProgramGenerated::coloredShadows()
{
    string shadowColored = R"(
    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        if (u_lightIsOn[i])
        {
            if (u_lightPosVS[i].w == 0.0)
            {
                vec3 S = normalize(-u_lightSpotDir[i].xyz);

                // Test if the current fragment is in shadow
                float shadow = u_matGetsShadows ? shadowTest(i, N, S) : 0.0;
                
                o_fragColor.rgb += shadow * indexToColor(getCascadesDepthIndex(i));
            }
        }
    }
    )";
    return shadowColored;
}

//-----------------------------------------------------------------------------

//! Adds the core shadow mapping test routine depending on the lights
string SLGLProgramGenerated::fragShadowTest(SLVLight* lights)
{
    int nbCascades = 0;

    for (SLLight* light : *lights)
    {
        if (light->doCascadedShadows())
        {
            nbCascades = light->shadowMap()->depthBuffers().size();
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


int getCascadesDepthIndex(in int i, int nbCascades)
{
    float ni = u_camNearPlane;
    float fi = u_camNearPlane;
    float factor = 30.0f;

    for (int i = 0; i < nbCascades-1; i++)
    {
        ni = fi;
        fi = factor * u_camNearPlane * pow((u_camFarPlane/(factor*u_camNearPlane)), float(i+1)/float(nbCascades));
        if (-v_P_VS.z < fi)
            return i;
    }
    return nbCascades-1;
}

float shadowTest(in int i, in vec3 N, in vec3 lightDir)
{
    if (u_lightCreatesShadows[i])
    {
        // Calculate position in light space
        mat4 lightSpace;
        vec3 lightToFragment = v_P_WS - u_lightPosWS[i].xyz;
)";

    if (nbCascades > 0)
    {
        shadowTestCode += R"(

        int index = 0;

        if (u_lightNbCascades[i] > 0)
        {
            index = getCascadesDepthIndex(i, u_lightNbCascades[i]);)";
        for (SLuint i = 0; i < lights->size(); ++i)
        {
            SLShadowMap* shadowMap = lights->at(i)->shadowMap();
            if (shadowMap && shadowMap->useCascaded())
                shadowTestCode += "if (i == " + std::to_string(i) + ") lightSpace = u_lightSpace_" + std::to_string(i) + "[index];\n";
        }
        shadowTestCode += R"(
        }
        else if (u_lightUsesCubemap[i])
        {)";
        for (SLuint i = 0; i < lights->size(); ++i)
        {
            SLShadowMap* shadowMap = lights->at(i)->shadowMap();
            if (shadowMap && shadowMap->useCubemap())
                shadowTestCode += "if (i == " + std::to_string(i) + ") lightSpace = u_lightSpace_" + std::to_string(i) + "[vectorToFace(lightToFragment)];\n";
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
            shadowTestCode += "if (i == " + std::to_string(i) + ") lightSpace = u_lightSpace_" + std::to_string(i); + "\n";
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
                shadowTestCode += "if (i == " + std::to_string(i) + ") lightSpace = u_lightSpace_" + std::to_string(i) + "[vectorToFace(lightToFragment)];\n";
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
                shadowTestCode += "if (i == " + std::to_string(i) + ") lightSpace = u_lightSpace_" + std::to_string(i) + ";\n";
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
            shadowTestCode += "            if (i == " + to_string(i) + ") texelSize = 1.0 / vec2(textureSize(u_shadowMap_" + to_string(i) + ", 0));\n";
        else if (shadowMap->useCascaded())
            shadowTestCode += "            if (i == " + to_string(i) + ") texelSize = 1.0 / vec2(textureSize(u_cascadedShadowMap_" + to_string(i) + "[index]" + ", 0));\n";
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
                shadowTestCode += "            if (i == " + to_string(i) + ") closestDepth = texture(u_shadowMap_" + to_string(i) + ", projCoords.xy + vec2(x, y) * texelSize).r;\n";
            else if (shadowMap->useCascaded())
                shadowTestCode += "            if (i == " + to_string(i) + ") closestDepth = texture(u_cascadedShadowMap_" + to_string(i) + "[index]" + ", projCoords.xy + vec2(x, y) * texelSize).r;\n";
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
            SLShadowMap* shadowMap =  lights->at(i)->shadowMap();
            if (shadowMap->useCubemap())
                shadowTestCode += "                if (i == " + to_string(i) + ") closestDepth = texture(u_shadowMapCube_" + to_string(i) + ", lightToFragment).r;\n";
    }
    shadowTestCode += R"(
            }
            else if (u_lightNbCascades[i] > 0)
            {
)";
    for (SLuint i = 0; i < lights->size(); ++i)
    {
        SLLight* light = lights->at(i);
        if (light->doCascadedShadows())
        {
            shadowTestCode += "                if (i == " + to_string(i) + ") { closestDepth = texture(u_cascadedShadowMap_" + to_string(i) + "[index]" + ", projCoords.xy).r; }\n";
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
            shadowTestCode += "                if (i == " + to_string(i) + ") closestDepth = texture(u_shadowMap_" + to_string(i) + ", projCoords.xy).r;\n"; }

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
    shader->code(SLGLShader::removeComments(code));
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
string SLGLProgramGenerated::shaderHeader(int numLights, int numCascades)

{
    string header = "\nprecision highp float;\n";
    header += "\n#define NUM_LIGHTS " + to_string(numLights) + "\n";
    if (numCascades != 0)
        header += "\n#define NUM_CASCADES " + to_string(numCascades) + "\n";
    return header;
}

//-----------------------------------------------------------------------------
