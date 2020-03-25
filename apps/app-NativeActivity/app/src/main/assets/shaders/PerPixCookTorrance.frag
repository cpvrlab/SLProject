//#############################################################################
//  File:      PerPixCookTorrance.frag
//  Purpose:   GLSL fragment shader for Cook-Torrance physical based rendering.
//             Based on the physically based rendering (PBR) tutorial with GLSL
//             from Joey de Vries on https://learnopengl.com/#!PBR/Theory
//  Author:    Marcus Hudritsch
//  Date:      July 2017
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision highp float;
#endif

varying vec3   v_P_VS;              //!< Interpol. point of illum. in view space (VS)
varying vec3   v_N_VS;              //!< Interpol. normal at v_P_VS in view space

uniform int    u_numLightsUsed;     //!< NO. of lights used light arrays
uniform bool   u_lightIsOn[8];      //!< flag if light is on
uniform vec4   u_lightPosVS[8];     //!< position of light in view space
uniform vec4   u_lightDiffuse[8];   //!< diffuse light intensity (Id)

uniform vec4   u_matDiffuse;        //!< diffuse color reflection coefficient (kd)
uniform float  u_matRoughness;      //!< Cook-Torrance material roughness 0-1
uniform float  u_matMetallic;       //!< Cook-Torrance material metallic 0-1

uniform float  u_oneOverGamma;      // 1.0f / Gamma correction value

const float AO = 1.0;               //!< Constant ambient occlusion factor
const float PI = 3.14159265359;
//-----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
//-----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
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
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
//-----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
//-----------------------------------------------------------------------------
void PointLight (in    int   i,         // Light number
                 in    vec3  P_VS,      // Point of illumination in VS
                 in    vec3  N,         // Normalized normal at v_P_VS
                 in    vec3  V,         // Normalized vector from v_P_VS to view in VS
                 in    vec3  F0,        // Frenel reflection at 90 deg. (0 to N)
                 in    vec3  diffuse,   // Material diffuse color
                 in    float roughness, // Material roughness
                 in    float metallic,  // Material metallic
                 inout vec3  Lo)        // reflected intensity
{
    vec3 L = u_lightPosVS[i].xyz - v_P_VS;      // Vector from v_P_VS to the light in VS
    float distance = length(L);                 // distance to light
    L /= distance;                              // normalize light vector
    vec3 H = normalize(V + L);                  // Normalized halfvector between eye and light vector
    float att = 1.0 / (distance*distance);      // quadratic light attenuation
    vec3 radiance = u_lightDiffuse[i].rgb * att;// per light radiance

     // cook-torrance brdf
     float NDF = DistributionGGX(N, H, roughness);
     float G   = GeometrySmith(N, V, L, roughness);
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
void main()
{
    vec3 N = normalize(v_N_VS);  // A varying normal has not anymore unit length
    vec3 V = normalize(-v_P_VS); // Vector from p to the viewer
    vec3 F0 = vec3(0.04);        // Init Frenel reflection at 90 deg. (0 to N)
    F0 = mix(F0, u_matDiffuse.rgb, u_matMetallic);

    // Get the reflection from all lights into Lo
    vec3 Lo = vec3(0.0);
    if (u_lightIsOn[0]) {PointLight(0, v_P_VS, N, V, F0, u_matDiffuse.rgb, u_matRoughness, u_matMetallic, Lo);}
    if (u_lightIsOn[1]) {PointLight(1, v_P_VS, N, V, F0, u_matDiffuse.rgb, u_matRoughness, u_matMetallic, Lo);}
    if (u_lightIsOn[2]) {PointLight(2, v_P_VS, N, V, F0, u_matDiffuse.rgb, u_matRoughness, u_matMetallic, Lo);}
    if (u_lightIsOn[3]) {PointLight(3, v_P_VS, N, V, F0, u_matDiffuse.rgb, u_matRoughness, u_matMetallic, Lo);}
    if (u_lightIsOn[4]) {PointLight(4, v_P_VS, N, V, F0, u_matDiffuse.rgb, u_matRoughness, u_matMetallic, Lo);}
    if (u_lightIsOn[5]) {PointLight(5, v_P_VS, N, V, F0, u_matDiffuse.rgb, u_matRoughness, u_matMetallic, Lo);}
    if (u_lightIsOn[6]) {PointLight(6, v_P_VS, N, V, F0, u_matDiffuse.rgb, u_matRoughness, u_matMetallic, Lo);}
    if (u_lightIsOn[7]) {PointLight(7, v_P_VS, N, V, F0, u_matDiffuse.rgb, u_matRoughness, u_matMetallic, Lo);}

    // ambient lighting (note that the next IBL tutorial will replace
    // this ambient lighting with environment lighting).
    vec3 ambient = vec3(0.03) * u_matDiffuse.rgb * AO;
    vec3 color = ambient + Lo;

    // HDR tonemapping
    color = color / (color + vec3(1.0));

    // Apply gamma correction
    color.rgb = pow(color.rgb, vec3(u_oneOverGamma));

    // set the fragment color with opaque alpha
    gl_FragColor = vec4(color, 1.0);
}
//-----------------------------------------------------------------------------
