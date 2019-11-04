//#############################################################################
//  File:      CT-Tex.frag
//  Purpose:   Calculated direct illumination using Blinn-Phong
//             and indirect illumination using voxel cone tracing
//  Author:    Stefan Thoeni
//  Date:      September 2018
//  Copyright: Stefan Thoeni
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#version 430 core
in vec3 o_N_WS;
in vec3 o_P_VS;
in vec3 o_P_WS;
in vec2 o_Tc;

#define VOXEL_SIZE (1/64.0)
#define SQRT2 (1.41421)
#define SQRT3 (1.732050807)
#define SQRT3DOUBLE (2 * 1.732050807)

// general settings:
uniform float   s_diffuseConeAngle;
uniform float   s_specularConeAngle;
uniform bool    s_directEnabled;
uniform bool    s_diffuseEnabled;
uniform bool    s_specularEnabled;
uniform bool    s_shadowsEnabled;
uniform float   s_shadowConeAngle;

// how big the mesh of a lightsource is. (This can vary from scene to scene)
uniform float   s_lightMeshSize;

// camera settings:
uniform vec3 u_EyePos;
uniform vec3 u_EyePosWS;

// Material & Light settings:
uniform int    u_numLightsUsed;     // NO. of lights used light arrays
uniform bool   u_lightIsOn[8];      // flag if light is on
uniform vec4   u_lightPosVS[8];     // position of light in voxel space
uniform vec4   u_lightPosWS[8];     // position of light in world space
uniform vec4   u_lightAmbient[8];   // ambient light intensity (Ia)
uniform vec4   u_lightDiffuse[8];   // diffuse light intensity (Id)
uniform vec4   u_lightSpecular[8];  // specular light intensity (Is)
uniform vec3   u_lightSpotDirWS[8]; // spot direction in world space
uniform float  u_lightSpotCutoff[8];// spot cutoff angle 1-180 degrees
uniform float  u_lightSpotCosCut[8];// cosine of spot cutoff angle
uniform float  u_lightSpotExp[8];   // spot exponent
uniform vec3   u_lightAtt[8];       // attenuation (const,linear,quadr.)
uniform bool   u_lightDoAtt[8];     // flag if att. must be calc.
uniform vec4   u_globalAmbient;     // Global ambient scene color

uniform vec4   u_matAmbient;        // ambient color reflection coefficient (ka)
uniform vec4   u_matDiffuse;        // diffuse color reflection coefficient (kd)
uniform vec4   u_matSpecular;       // specular color reflection coefficient (ks)
uniform vec4   u_matEmissive;       // emissive color for selfshining materials
uniform float  u_matShininess;      // shininess exponent
uniform float  u_matKr;             // reflection factor (kr)
uniform bool   u_matHasTexture;     // flag if material has texture
uniform float  u_oneOverGamma;		// oneOverGamma correction factor

uniform sampler2D u_texture0;       // Color texture map
uniform sampler3D u_texture3D;        // Voxelization texture.

out vec4 color;

//-----------------------------------------------------------------------------
// Returns true if the point p is inside the unity cube.
bool isInsideCube(const vec3 p, float e) 
{ 
    return abs(p.x) < 1 + e && 
           abs(p.y) < 1 + e && 
           abs(p.z) < 1 + e; 
}
//-----------------------------------------------------------------------------
vec4 coneTraceStopDist(vec3  from, 
                       vec3  dir, 
                       float angle, 
                       float stopDistance, 
                       vec3  offsetVector, 
                       float firstSampleDist)
{
    dir = normalize(dir);

    vec3  res = vec3(0.0f);
    float alpha = 0.0;
    float dist = VOXEL_SIZE * firstSampleDist;

    // offset to avoid sampling its own voxel
    vec3 offset = normalize(offsetVector) * VOXEL_SIZE * SQRT3;
    from = from + offset;

    float tanTheta2 = tan(angle) * 2;

    stopDistance -= length(offset); // remove the offset from distance
  
    while(dist < stopDistance && alpha < 1) 
    {
        // calculate voxel coordinate:
        vec3 coordinate = from + dist * dir;

        //if(!isInsideCube(coordinate, 0.0)) break;
        
        float diameter = max(VOXEL_SIZE, tanTheta2 * dist);
        float mip = log2(diameter / VOXEL_SIZE);
        if(mip > 6) break;
        vec4 samp = textureLod(u_texture3D, coordinate, mip);

        // alpha blending
        float f = 1 - alpha;
        res.rgb += samp.rgb * f; 
        alpha += samp.a * f;
    
        dist += diameter * 0.55;
    }
    return vec4(res, alpha);
}
//-----------------------------------------------------------------------------
vec4 coneTrace(vec3 from, vec3 dir, float angle)
{
    const float stopDistance = 50 * VOXEL_SIZE; // performance boost!
    return coneTraceStopDist(from, dir, angle, stopDistance, o_N_WS, 3.0);
}
//-----------------------------------------------------------------------------
void DirectLight(in    int  i,   // Light number
                 in    vec3 N,   // Normalized normal 
                 in    vec3 E,   // Normalized vector 
                 inout vec4 Id,  // Diffuse light intesity
                 inout vec4 Is)  // Specular light intesity
{  
    // We use the spot light direction as the light direction vector
    vec3 L = normalize(-u_lightSpotDirWS[i].xyz);

    // Half vector H between L and E
    vec3 H = normalize(L+E);
   
    // Calculate diffuse & specular factors
    float diffFactor = max(dot(N,L), 0.0);
    float specFactor = 0.0;
    if (diffFactor!=0.0) 
    specFactor = pow(max(dot(N,H), 0.0), u_matShininess);
   
    // accumulate directional light intesities w/o attenuation
    Id += u_lightDiffuse[i] * diffFactor;
    Is += u_lightSpecular[i] * specFactor;
}
//-----------------------------------------------------------------------------
void PointLight (in    int  i,      // Light number
                 in    vec3 P_WS,   // Point of illumination
                 in    vec3 N,      // Normalized normal 
                 in    vec3 E,      // Normalized eye vector
                 inout vec4 Id,     // Diffuse light intensity
                 inout vec4 Is)     // Specular light intensity
{
    // Vector from P_VS to the light in VS
    vec3 L = u_lightPosWS[i].xyz - P_WS;
      
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
   
    // Normalized halfvector between N and L
    vec3 H = normalize(L+E);
   
    // Calculate diffuse & specular factors
    float diffFactor = max(dot(N,L), 0.0);
    float specFactor = 0.0;
    if (diffFactor!=0.0) 
        specFactor = pow(max(dot(N,H), 0.0), u_matShininess);
   
    // Calculate spot attenuation
    if (u_lightSpotCutoff[i] < 180.0)
    {   float spotDot; // Cosine of angle between L and spotdir
        float spotAtt; // Spot attenuation
        spotDot = dot(-L, u_lightSpotDirWS[i]);
        if (spotDot < u_lightSpotCosCut[i]) spotAtt = 0.0;
        else spotAtt = max(pow(spotDot, u_lightSpotExp[i]), 0.0);
        att *= spotAtt;
    }

    vec4 shadow = vec4(0.0);

    if(s_shadowsEnabled)
    {
        // Calculate shadow: offset along normal just a teeny tiny bit. 
        // This improves shadow quality by removing ugly self-shadowing artefacts:
        vec3 from = o_P_VS + (normalize(o_N_WS) * VOXEL_SIZE);

        // Directional lights come with a mesh that occupies the same voxel as the light.
        // For that reason, we offset the light away from this voxel. 
        // This of corse offsets the shadows and introduces a small error
        vec3 to = (u_lightPosVS[0].xyz + normalize(u_lightSpotDirWS[i]) * VOXEL_SIZE * SQRT3) - from;

        // Stop sampling for shadows 1 voxel diagonal before reaching the light,
        // again to avoid sampling a lights mesh.
        shadow = coneTraceStopDist(from, 
                                   to, 
                                   s_shadowConeAngle, 
                                   length(to) - VOXEL_SIZE * SQRT3 * s_lightMeshSize, 
                                   to, 
                                   1);
    }
    
    // Accumulate light intesities
    Id += att * u_lightDiffuse[i]  * diffFactor * (1 - shadow.a);
    Is += att * u_lightSpecular[i] * specFactor * (1 - shadow.a);
}
//-----------------------------------------------------------------------------
vec4 direct()
{
    vec4 Id, Is;        // Accumulated light intensities at v_P_VS
   
    Id = vec4(0.0);     // Diffuse light intesity
    Is = vec4(0.0);     // Specular light intesity
   
    vec3 N = normalize(o_N_WS);  // A varying normal has not anymore unit length
    vec3 E = normalize(u_EyePosWS - o_P_WS); // Vector from p to the eye

    if (u_lightIsOn[0]) {if (u_lightPosVS[0].w == 0.0) DirectLight(0, N, E, Id, Is); else PointLight(0, o_P_WS, N, E, Id, Is);}
    if (u_lightIsOn[1]) {if (u_lightPosVS[1].w == 0.0) DirectLight(1, N, E, Id, Is); else PointLight(1, o_P_WS, N, E, Id, Is);}
    if (u_lightIsOn[2]) {if (u_lightPosVS[2].w == 0.0) DirectLight(2, N, E, Id, Is); else PointLight(2, o_P_WS, N, E, Id, Is);}
    if (u_lightIsOn[3]) {if (u_lightPosVS[3].w == 0.0) DirectLight(3, N, E, Id, Is); else PointLight(3, o_P_WS, N, E, Id, Is);}
    if (u_lightIsOn[4]) {if (u_lightPosVS[4].w == 0.0) DirectLight(4, N, E, Id, Is); else PointLight(4, o_P_WS, N, E, Id, Is);}
    if (u_lightIsOn[5]) {if (u_lightPosVS[5].w == 0.0) DirectLight(5, N, E, Id, Is); else PointLight(5, o_P_WS, N, E, Id, Is);}
    if (u_lightIsOn[6]) {if (u_lightPosVS[6].w == 0.0) DirectLight(6, N, E, Id, Is); else PointLight(6, o_P_WS, N, E, Id, Is);}
    if (u_lightIsOn[7]) {if (u_lightPosVS[7].w == 0.0) DirectLight(7, N, E, Id, Is); else PointLight(7, o_P_WS, N, E, Id, Is);}
     
    // Sum up all the direct reflected color components
    vec4 color =  u_matEmissive +
                  Id * u_matDiffuse;
               
    // Componentwise multiply w. texture color
    color *= texture2D(u_texture0, o_Tc); 
    
    // add finally the specular RGB-part
    vec4 specColor = Is * u_matSpecular;
    color.rgb += specColor.rgb;
    return color;
}
//-----------------------------------------------------------------------------
vec4 indirectDiffuse()
{
    vec4 res = vec4(0.0);
    vec3 N = normalize(o_N_WS);

    // a vector orthogonal to N:
    vec3 ortho1;
    if(N.z >= 0.9)
        ortho1 = normalize(vec3(0.0, -N.z, N.y));
    else
        ortho1 = normalize(vec3(-N.y, N.x, 0.0));

    // a vector orthogonal to N and ortho1
    vec3 ortho2 = normalize(cross(N, ortho1));
    
    const vec3 from = o_P_VS;

    // also offset the cone along its direction a bit:

    res += coneTrace(from , N, s_diffuseConeAngle);

    // Trace 4 side cones.
    float mixIn = 0.5;
    const vec3 s1 = mix(N,  ortho1, mixIn);
    const vec3 s2 = mix(N, -ortho1, mixIn);
    const vec3 s3 = mix(N,  ortho2, mixIn);
    const vec3 s4 = mix(N, -ortho2, mixIn);

    res +=  coneTrace(from, s1, s_diffuseConeAngle);
    res +=  coneTrace(from, s2, s_diffuseConeAngle);
    res +=  coneTrace(from, s3, s_diffuseConeAngle);
    res +=  coneTrace(from, s4, s_diffuseConeAngle);

    const vec3 corner1 = 0.5f * (ortho1 + ortho2);
    const vec3 corner2 = 0.5f * (ortho1 - ortho2);

    const vec3 c1 = mix(N,  corner1, mixIn);
    const vec3 c2 = mix(N, -corner1, mixIn);
    const vec3 c3 = mix(N,  corner2, mixIn);
    const vec3 c4 = mix(N, -corner2, mixIn);

    res +=  coneTrace(from, c1, s_diffuseConeAngle);
    res +=  coneTrace(from, c2, s_diffuseConeAngle);
    res +=  coneTrace(from, c3, s_diffuseConeAngle);
    res +=  coneTrace(from, c4, s_diffuseConeAngle);

    return (res / 9) * u_matDiffuse;
}
//-----------------------------------------------------------------------------
vec4 indirectSpecularLight()
{
    vec3 N = normalize(o_N_WS);
    vec3 E = normalize(u_EyePos - o_P_VS);
    vec3 R = normalize(reflect(-E, N));

    vec4 spec = coneTraceStopDist(o_P_VS, 
                                  R, 
                                  s_specularConeAngle, 
                                  SQRT3DOUBLE, 
                                  o_N_WS, 
                                  0.0);

    return u_matKr * u_matSpecular * spec;
}
//-----------------------------------------------------------------------------
void main()
{
    color = vec4(0, 0, 0, 1);

    if(s_diffuseEnabled)
        color += indirectDiffuse();

    if(s_directEnabled)
        color += direct();
  
    if(s_specularEnabled)
        color += indirectSpecularLight();

    color.rgb = pow(color.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
