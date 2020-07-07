//#############################################################################
//  File:      CTVoxelization.frag
//  Purpose:   Calculates diffuse illum. & stores it into voxelized 3D texture
//  Author:    Stefan Thoeni
//  Date:      September 2018
//  Copyright: Stefan Thoeni
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#version 430 core
//-----------------------------------------------------------------------------
in vec3 o_F_WS;
in vec3 o_F_VS;
in vec3 o_N_WS;

uniform vec3 u_EyePos;              // camera settings:

uniform int    u_numLightsUsed;     // NO. of lights used light arrays
uniform bool   u_lightIsOn[8];      // flag if light is on
uniform vec4   u_lightPosWS[8];     // position of light in world space
uniform vec4   u_lightAmbi[8];   // ambient light intensity (Ia)
uniform vec4   u_lightDiff[8];   // diffuse light intensity (Id)
uniform vec4   u_lightSpec[8];  // specular light intensity (Is)
uniform vec3   u_lightSpotDirWS[8]; // spot direction in view space
uniform float  u_lightSpotDeg[8];// spot cutoff angle 1-180 degrees
uniform float  u_lightSpotCos[8];// cosine of spot cutoff angle
uniform float  u_lightSpotExp[8];   // spot exponente
uniform vec3   u_lightAtt[8];       // attenuation (const,linear,quadr.)
uniform bool   u_lightDoAtt[8];     // flag if att. must be calc.
uniform vec4   u_globalAmbi;     // Global ambient scene color

uniform vec4   u_matAmbi;        // ambient color reflection coefficient (ka)
uniform vec4   u_matDiff;        // diffuse color reflection coefficient (kd)
uniform vec4   u_matSpec;       // specular color reflection coefficient (ks)
uniform vec4   u_matEmis;       // emissive color for self-shining materials
uniform float  u_matShin;      // shininess exponent

layout(RGBA8) uniform image3D texture3D;
//-----------------------------------------------------------------------------
void DirectLight(in    int  i,   // Light number
                 in    vec3 N,   // Normalized normal 
                 in    vec3 E,   // Normalized vector 
                 inout vec4 Ia,  // Ambient light intensity
                 inout vec4 Id,  // Diffuse light intensity
                 inout vec4 Is)  // Specular light intensity
{  
    // We use the spot light direction as the light direction vector
    vec3 L = normalize(-u_lightSpotDirWS[i].xyz);

    // Half vector H between L and E
    vec3 H = normalize(L+E);
   
    // Calculate diffuse & specular factors
    float diffFactor = max(dot(N,L), 0.0);
    float specFactor = 0.0;
    if (diffFactor!=0.0) 
        specFactor = pow(max(dot(N,H), 0.0), u_matShin);
   
    // accumulate directional light intesities w/o attenuation
    Ia += u_lightAmbi[i];
    Id += u_lightDiff[i] * diffFactor;
    Is += u_lightSpec[i] * specFactor;
}
//-----------------------------------------------------------------------------
void PointLight (in    int  i,      // Light number
                 in    vec3 P_WS,   // Point of illumination
                 in    vec3 N,      // Normalized normal 
                 in    vec3 E,      // Normalized eye vector
                 inout vec4 Ia,     // Ambient light intensity
                 inout vec4 Id,     // Diffuse light intensity
                 inout vec4 Is)     // Specular light intensity
{  
    // Vector from v_P_VS to the light in VS
  vec3 L = u_lightPosWS[i].xyz - o_F_WS;
   
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
        specFactor = pow(max(dot(N,H), 0.0), u_matShin);
   
    // Calculate spot attenuation
    if (u_lightSpotDeg[i] < 180.0)
    {   float spotDot; // Cosine of angle between L and spotdir
        float spotAtt; // Spot attenuation
        spotDot = dot(-L, u_lightSpotDirWS[i]);
        if (spotDot < u_lightSpotCos[i]) spotAtt = 0.0;
        else spotAtt = max(pow(spotDot, u_lightSpotExp[i]), 0.0);
        att *= spotAtt;
    }
   
    // Accumulate light intesities
    Id += att * u_lightDiff[i] * diffFactor;

}

void main(){
  vec4 Ia, Id, Is;        // Accumulated light intensities at v_P_VS
   
  Ia = vec4(0.0);         // Ambient light intensity
  Id = vec4(0.0);         // Diffuse light intensity
  Is = vec4(0.0);         // Specular light intensity
   
  vec3 N = normalize(o_N_WS);  // A input normal has not anymore unit length
  vec3 E = normalize(u_EyePos - o_F_WS); // Vector from p to the eye

  if (u_lightIsOn[0]) {if (u_lightPosWS[0].w == 0.0) DirectLight(0, N, E, Ia, Id, Is); else PointLight(0, o_F_WS, N, E, Ia, Id, Is);}
  if (u_lightIsOn[1]) {if (u_lightPosWS[1].w == 0.0) DirectLight(1, N, E, Ia, Id, Is); else PointLight(1, o_F_WS, N, E, Ia, Id, Is);}
  if (u_lightIsOn[2]) {if (u_lightPosWS[2].w == 0.0) DirectLight(2, N, E, Ia, Id, Is); else PointLight(2, o_F_WS, N, E, Ia, Id, Is);}
  if (u_lightIsOn[3]) {if (u_lightPosWS[3].w == 0.0) DirectLight(3, N, E, Ia, Id, Is); else PointLight(3, o_F_WS, N, E, Ia, Id, Is);}
  if (u_lightIsOn[4]) {if (u_lightPosWS[4].w == 0.0) DirectLight(4, N, E, Ia, Id, Is); else PointLight(4, o_F_WS, N, E, Ia, Id, Is);}
  if (u_lightIsOn[5]) {if (u_lightPosWS[5].w == 0.0) DirectLight(5, N, E, Ia, Id, Is); else PointLight(5, o_F_WS, N, E, Ia, Id, Is);}
  if (u_lightIsOn[6]) {if (u_lightPosWS[6].w == 0.0) DirectLight(6, N, E, Ia, Id, Is); else PointLight(6, o_F_WS, N, E, Ia, Id, Is);}
  if (u_lightIsOn[7]) {if (u_lightPosWS[7].w == 0.0) DirectLight(7, N, E, Ia, Id, Is); else PointLight(7, o_F_WS, N, E, Ia, Id, Is);}

  vec4 color = u_matEmis +
               Ia * u_matAmbi +
               Id * u_matDiff +
               Is * u_matSpec;
  
  // Output lighting to 3D texture.
  ivec3 dim = imageSize(texture3D);

  vec4 res = vec4(vec3(color.xyz), 1);
  imageStore(texture3D,  ivec3(dim * o_F_VS), res);
}
//-----------------------------------------------------------------------------
