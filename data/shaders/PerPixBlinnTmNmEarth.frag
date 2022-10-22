//#############################################################################
//  File:      PerPixBlinnTmNmEarth.frag
//  Purpose:   OGLSL parallax bump mapping
//  Date:      July 2014
//  Authors:   Markus Knecht, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces #Lights by SLVLights.size()
#pragma define NUM_LIGHTS #Lights
//-----------------------------------------------------------------------------
in      vec3        v_P_VS;                     // Interpol. point of illumination in view space (VS)
in      vec2        v_uv0;                      // Texture coordinate varying
in      vec3        v_eyeDirTS;                 // Vector to the eye in tangent space
in      vec3        v_lightDirTS[NUM_LIGHTS];   // Vector to the light in tangent space
in      vec3        v_spotDirTS[NUM_LIGHTS];    // Spot direction in tangent space
in      float       v_lightDist[NUM_LIGHTS];    // Light distance

uniform bool        u_lightIsOn[NUM_LIGHTS];     // flag if light is on
uniform vec4        u_lightPosVS[NUM_LIGHTS];    // position of light in view space
uniform vec4        u_lightAmbi[NUM_LIGHTS];     // ambient light intensity (Ia)
uniform vec4        u_lightDiff[NUM_LIGHTS];     // diffuse light intensity (Id)
uniform vec4        u_lightSpec[NUM_LIGHTS];     // specular light intensity (Is)
uniform vec3        u_lightSpotDir[NUM_LIGHTS];  // spot direction in view space
uniform float       u_lightSpotDeg[NUM_LIGHTS];  // spot cutoff angle 1-180 degrees
uniform float       u_lightSpotCos[NUM_LIGHTS];  // cosine of spot cutoff angle
uniform float       u_lightSpotExp[NUM_LIGHTS];  // spot exponent
uniform vec3        u_lightAtt[NUM_LIGHTS];      // attenuation (const,linear,quadr.)
uniform bool        u_lightDoAtt[NUM_LIGHTS];    // flag if att. must be calc.
uniform vec4        u_globalAmbi;                // Global ambient scene color
uniform float       u_oneOverGamma;              // 1.0f / Gamma correction value

uniform vec4        u_matAmbi;              // ambient color reflection coefficient (ka)
uniform vec4        u_matDiff;              // diffuse color reflection coefficient (kd)
uniform vec4        u_matSpec;              // specular color reflection coefficient (ks)
uniform vec4        u_matEmis;              // emissive color for self-shining materials
uniform float       u_matShin;              // shininess exponent
uniform sampler2D   u_matTextureDiffuse0;   // Color map
uniform sampler2D   u_matTextureNormal0;    // Normal map
uniform sampler2D   u_matTextureHeight0;    // Height map;
uniform sampler2D   u_matTextureSpecular0;  // Gloss map;
uniform sampler2D   u_matTextureDiffuse1;   // Cloud Color map;
uniform sampler2D   u_matTextureDiffuse2;   // Night Color map;
uniform float       u_scale;                // Height scale for parallax mapping
uniform float       u_offset;               // Height bias for parallax mapping

uniform int         u_camProjType;        // type of stereo
uniform int         u_camStereoEye;         // -1=left, 0=center, 1=right
uniform mat3        u_camStereoColors;      // color filter matrix
uniform bool        u_camFogIsOn;           // flag if fog is on
uniform int         u_camFogMode;           // 0=LINEAR, 1=EXP, 2=EXP2
uniform float       u_camFogDensity;        // fog density value
uniform float       u_camFogStart;          // fog start distance
uniform float       u_camFogEnd;            // fog end distance
uniform vec4        u_camFogColor;          // fog color (usually the background)

out     vec4        o_fragColor;            // output fragment color
//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces the include pragma by the file
#pragma include "lightingBlinnPhong.glsl"
#pragma include "fogBlend.glsl"
#pragma include "doStereoSeparation.glsl"
//-----------------------------------------------------------------------------
void main()
{
    // Normalize E and L
    vec3 E = normalize(v_eyeDirTS);         // normalized eye direction
    vec3 L = normalize(v_lightDirTS[0]);    // normalized light direction
    vec3 S;                                 // normalized spot direction
   
    // Halfvector H between L & E (See Blinn's lighting model)
    vec3 H = normalize(L + E);
   
    ////////////////////////////////////////////////////////////
    // Calculate new texture coord. Tc for Parallax mapping
    // The height comes from red channel from the height map
    float height = texture(u_matTextureHeight0, v_uv0.st).r;
   
    // Scale the height and add the bias (height offset)
    height = height * u_scale + u_offset;
   
    // Add the texture offset to the texture coord.
    vec2 Tc = v_uv0.st + (height * E.st);

    // set clouds cord
    vec2 Wtc = v_uv0.st;
    ////////////////////////////////////////////////////////////
   
    // Get normal from normal map, move from [0,1] to [-1, 1] range & normalize
    vec3 N = normalize(texture(u_matTextureNormal0, Tc).rgb * 2.0 - 1.0);
   
    // Calculate attenuation over distance v_d
    float att = 1.0;  // Total attenuation factor
    if (u_lightDoAtt[0])
    {   vec3 att_dist;
        att_dist.x = 1.0;
        att_dist.y = v_lightDist[0];
        att_dist.z = v_lightDist[0] * v_lightDist[0];
        att = 1.0 / dot(att_dist, u_lightAtt[0]);
    }
   
    // Calculate spot attenuation
    if (u_lightSpotDeg[0] < 180.0)
    {   S = normalize(v_spotDirTS[0]);
        float spotDot; // Cosine of angle between L and spotdir
        float spotAtt; // Spot attenuation
        spotDot = dot(-L, S);
        if (spotDot < u_lightSpotCos[0]) spotAtt = 0.0;
        else spotAtt = max(pow(spotDot, u_lightSpotExp[0]), 0.0);
        att *= spotAtt;
    }   
   
    // compute diffuse lighting
    float diffFactor = max(dot(L,N), 0.0) ;
   
    // compute specular lighting
    float specFactor = pow(max(dot(N,H), 0.0), u_matShin);
   
    // add ambient & diffuse light components
    o_fragColor = u_matEmis +
                   u_globalAmbi +
                   att * u_lightAmbi[0] * u_matAmbi +
                   att * u_lightDiff[0] * u_matDiff * diffFactor;
   
    //Calculate day night limit
    float night = max(dot(L,vec3(0,0,1)),0.0);
   
    //Make day light limit smoother
    float nightInv = 1.0 - night;
    float night2 = nightInv * nightInv;
   
    //Calculate mixed day night texture 
    vec4 cloudColor = texture(u_matTextureDiffuse1, v_uv0.st);
    float cloudAlpha = cloudColor.a;
    vec4 ground = (texture(u_matTextureDiffuse2, Tc)*night2 +
                   texture(u_matTextureDiffuse0, Tc)*(1.0-night2))*(1.0-cloudAlpha);
   
    //Calculate CloudTexture
    vec4 cloud = o_fragColor*texture(u_matTextureDiffuse1, Wtc)*(cloudAlpha);
   
    o_fragColor = ground  + cloud;
   
    //Finally Add specular light
    o_fragColor += att *
                   u_lightSpec[0] *
                   u_matSpec *
                   specFactor *
                   texture(u_matTextureSpecular0, Tc)[0];

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));

    // Apply fog by blending over distance
    if (u_camFogIsOn)
        o_fragColor = fogBlend(v_P_VS, o_fragColor);

    // Apply stereo eye separation
    if (u_camProjType > 1)
        doStereoSeparation();
}
//-----------------------------------------------------------------------------
