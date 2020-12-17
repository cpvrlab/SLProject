//#############################################################################
//  File:      SLGLProgramGenerated.cpp
//  Author:    Marcus Hudritsch
//  Date:      December 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLApplication.h>
#include <SLAssetManager.h>
#include <SLGLProgramGenerated.h>
#include <SLGLShader.h>
#include <SLCamera.h>
#include <SLLight.h>

using std::string;

//-----------------------------------------------------------------------------
string directLightBlinnPhong = R"(
void directLightBlinnPhong(in    int  i,       // Light number between 0 and NUM_LIGHTS
                           in    vec3 N,       // Normalized normal at v_P
                           in    vec3 E,       // Normalized direction at v_P to the eye
                           in    vec3 S,       // Normalized light spot direction
                           in    float shadow, // shadow factor
                           inout vec4 Ia,      // Ambient light intensity
                           inout vec4 Id,      // Diffuse light intensity
                           inout vec4 Is)      // Specular light intensity
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
)";
//-----------------------------------------------------------------------------
string pointLightBlinnPhong = R"(
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

    // Accumulate light intesities
    Ia += att * u_lightAmbi[i];
    Id += att * u_lightDiff[i] * diffFactor * (1.0 - shadow);
    Is += att * u_lightSpec[i] * specFactor * (1.0 - shadow);
}
)";
//-----------------------------------------------------------------------------
string directLightCookTorrance = R"(

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
)";
//-----------------------------------------------------------------------------
string pointLightCookTorrance = R"(

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
string doStereoSeparation = R"(
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
string fogBlend = R"(
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
string stringName4 = R"(

)";
//-----------------------------------------------------------------------------
string stringName5 = R"(

)";
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildShaderProgram(SLMaterial* mat,
                                              SLCamera*   cam,
                                              SLVLight*   lights)
{
    assert(mat && "No material pointer passed!");
    assert(cam && "No camera pointer passed!");
    assert(!lights->empty() && "No lights passed!");

    bool matHasTm = !mat->textures().empty();
    bool matHasNm = mat->textures().size() > 1 &&
                    mat->textures()[1]->texType() == TT_normal;
    bool matHasAo = mat->textures().size() > 2 &&
                    mat->textures()[2]->texType() == TT_ambientOcclusion;
    bool lightsHaveSm = lights->at(0)->createsShadows();

    if (mat->lightModel() == LM_BlinnPhong)
    {
        if (matHasTm)
        {
            if (matHasNm && matHasAo && lightsHaveSm)
                buildPerPixBlinnTmNmAoSm(mat, cam, lights);
            else if (matHasNm && matHasAo)
                buildPerPixBlinnTmNmAo(mat, cam, lights);
            else if (matHasNm && lightsHaveSm)
                buildPerPixBlinnTmNmSm(mat, cam, lights);
            else if (matHasNm)
                buildPerPixBlinnTmNm(mat, cam, lights);
            else if (lightsHaveSm)
                buildPerPixBlinnTmSm(mat, cam, lights);
            else
                buildPerPixBlinnTm(mat, cam, lights);
        }
        else
        {
            if (matHasAo && lightsHaveSm)
                buildPerPixBlinnAoSm(mat, cam, lights);
            else if (lightsHaveSm)
                buildPerPixBlinnSm(mat, cam, lights);
            else
                buildPerPixBlinn(mat, cam, lights);
        }
    }
    else
        SL_EXIT_MSG("Only Blinn-Phong supported yet.");
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTmNmAoSm(SLMaterial* mat,
                                                    SLCamera*   cam,
                                                    SLVLight*   lights)
{
    assert(_shaders.size() > 1 &&
           _shaders[0]->type() == ST_vertex &&
           _shaders[1]->type() == ST_fragment);

    SLGLShader* vrtSh = _shaders[0];
    SLGLShader* frgSh = _shaders[1];

    string code;
    code += "precision highp float;\n";
    code += "#define NUM_LIGHTS " + std::to_string(lights->size()) + "\n";
    code += R"(

layout (location = 0) in vec4  a_position;  // Vertex position attribute
layout (location = 1) in vec3  a_normal;    // Vertex normal attribute
layout (location = 2) in vec2  a_uv1;       // Vertex tex.coord. 1 for diffuse color
layout (location = 3) in vec2  a_uv2;       // Vertex tex.coord. 2 for AO
layout (location = 5) in vec4  a_tangent;   // Vertex tangent attribute

uniform mat4  u_mvMatrix;   // modelview matrix
uniform mat3  u_nMatrix;    // normal matrix=transpose(inverse(mv))
uniform mat4  u_mvpMatrix;  // = projection * modelView
uniform mat4  u_mMatrix;    // model matrix

uniform vec4  u_lightPosVS[NUM_LIGHTS];     // position of light in view space
uniform vec3  u_lightSpotDir[NUM_LIGHTS];   // spot direction in view space
uniform float u_lightSpotDeg[NUM_LIGHTS];   // spot cutoff angle 1-180 degrees

out     vec3  v_P_VS;                   // Point of illumination in view space (VS)
out     vec3  v_P_WS;                   // Point of illumination in world space (WS)
out     vec3  v_N_VS;                   // Normal at P_VS in view space
out     vec2  v_uv1;                    // Texture coordiante 1 output
out     vec2  v_uv2;                    // Texture coordiante 2 output
out     vec3  v_eyeDirTS;               // Vector to the eye in tangent space
out     vec3  v_lightDirTS[NUM_LIGHTS]; // Vector to the light 0 in tangent space
out     vec3  v_spotDirTS[NUM_LIGHTS];  // Spot direction in tangent space
//-----------------------------------------------------------------------------
void main()
{
    v_uv1 = a_uv1;  // pass diffuse color tex.coord. 1 for interpolation
    v_uv2 = a_uv2;  // pass ambient occlusion tex.coord. 2 for interpolation

    // Building the matrix Eye Space -> Tangent Space
    // See the math behind at: http://www.terathon.com/code/tangent.html
    vec3 n = normalize(u_nMatrix * a_normal);
    vec3 t = normalize(u_nMatrix * a_tangent.xyz);
    vec3 b = cross(n, t) * a_tangent.w; // bitangent w. corrected handedness
    mat3 TBN = mat3(t,b,n);

    v_P_VS = vec3(u_mvMatrix *  a_position); // vertex position in view space
    v_P_WS = vec3(u_mMatrix * a_position);   // vertex position in world space

    // Transform vector to the eye into tangent space
    v_eyeDirTS = -v_P_VS;  // eye vector in view space
    v_eyeDirTS *= TBN;

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        // Transform spotdir into tangent space
        v_spotDirTS[i] = u_lightSpotDir[i];
        v_spotDirTS[i]  *= TBN;

        // Transform vector to the light 0 into tangent space
        vec3 L = u_lightPosVS[i].xyz - v_P_VS;
        v_lightDirTS[i] = L;
        v_lightDirTS[i] *= TBN;
    }

    // pass the vertex w. the fix-function transform
    gl_Position = u_mvpMatrix * a_position;
}
)";

    vrtSh->code(code);
    vrtSh->name("generatedPerPixBlinnTmNmAoSm.vert");
    vrtSh->file(SLApplication::configPath + name());

}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTmNmAo(SLMaterial* mat,
                                                  SLCamera*   cam,
                                                  SLVLight*   lights)
{
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTmNmSm(SLMaterial* mat,
                                                  SLCamera*   cam,
                                                  SLVLight*   lights)
{
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTmNm(SLMaterial* mat,
                                                SLCamera*   cam,
                                                SLVLight*   lights)
{
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTmSm(SLMaterial* mat,
                                                SLCamera*   cam,
                                                SLVLight*   lights)
{
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnAoSm(SLMaterial* mat,
                                                SLCamera*   cam,
                                                SLVLight*   lights)
{
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnSm(SLMaterial* mat,
                                              SLCamera*   cam,
                                              SLVLight*   lights)
{
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinnTm(SLMaterial* mat,
                                              SLCamera*   cam,
                                              SLVLight*   lights)
{
}
//-----------------------------------------------------------------------------
void SLGLProgramGenerated::buildPerPixBlinn(SLMaterial* mat,
                                            SLCamera*   cam,
                                            SLVLight*   lights)
{
}
//-----------------------------------------------------------------------------
