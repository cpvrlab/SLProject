//#############################################################################
//  File:      PerPixBlinnNrm.frag
//  Purpose:   GLSL normal map bump mapping
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
in      vec3        v_P_VS;     // Interpol. point of illum. in view space (VS)
in      vec2        v_texCoord; // Texture coordiante varying
in      vec3        v_L_TS;     // Vector to light 0 in tangent space
in      vec3        v_E_TS;     // Vector to the eye in tangent space
in      vec3        v_S_TS;     // Spot direction in tangent space
in      float       v_d;        // Light distance

uniform bool        u_lightIsOn[NUM_LIGHTS];        // flag if light is on
uniform vec4        u_lightPosVS[NUM_LIGHTS];       // position of light in view space
uniform vec4        u_lightAmbient[NUM_LIGHTS];     // ambient light intensity (Ia)
uniform vec4        u_lightDiffuse[NUM_LIGHTS];     // diffuse light intensity (Id)
uniform vec4        u_lightSpecular[NUM_LIGHTS];    // specular light intensity (Is)
uniform vec3        u_lightSpotDirVS[NUM_LIGHTS];   // spot direction in view space
uniform float       u_lightSpotCutoff[NUM_LIGHTS];  // spot cutoff angle 1-180 degrees
uniform float       u_lightSpotCosCut[NUM_LIGHTS];  // cosine of spot cutoff angle
uniform float       u_lightSpotExp[NUM_LIGHTS];     // spot exponent
uniform vec3        u_lightAtt[NUM_LIGHTS];         // attenuation (const,linear,quadr.)
uniform bool        u_lightDoAtt[NUM_LIGHTS];       // flag if att. must be calc.
uniform vec4        u_globalAmbient;                // Global ambient scene color
uniform float       u_oneOverGamma;                 // 1.0f / Gamma correction value

uniform vec4        u_matAmbient;           // ambient color reflection coefficient (ka)
uniform vec4        u_matDiffuse;           // diffuse color reflection coefficient (kd)
uniform vec4        u_matSpecular;          // specular color reflection coefficient (ks)
uniform vec4        u_matEmissive;          // emissive color for self-shining materials
uniform float       u_matShininess;         // shininess exponent
uniform sampler2D   u_matTexture0;          // Color map
uniform sampler2D   u_matTexture1;          // Normal map

uniform int         u_camProjection;        // type of stereo
uniform int         u_camStereoEye;         // -1=left, 0=center, 1=right
uniform mat3        u_camStereoColors;      // color filter matrix
uniform bool        u_camFogIsOn;           // flag if fog is on
uniform int         u_camFogMode;           // 0=LINEAR, 1=EXP, 2=EXP2
uniform float       u_camFogDensity;        // fog densitiy value
uniform float       u_camFogStart;          // fog start distance
uniform float       u_camFogEnd;            // fog end distance
uniform vec4        u_camFogColor;          // fog color (usually the background)

out     vec4        o_fragColor;            // output fragment color
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
    vec3 E = normalize(v_E_TS);   // normalized eye direction
    vec3 L = normalize(v_L_TS);   // normalized light direction
   
    // Halfvector H between L & E (See Blinn's lighting model)
    vec3 H = normalize(L + E);
  
    // Get normal from normal map, move from [0,1] to [-1, 1] range & normalize
    vec3 N = normalize(texture(u_matTexture1, v_texCoord).rgb * 2.0 - 1.0);
   
    // Calculate attenuation over distance d
    float att = 1.0;  // Total attenuation factor
    if (u_lightDoAtt[0])
    {   vec3 att_dist;
        att_dist.x = 1.0;
        att_dist.y = v_d;
        att_dist.z = v_d * v_d;
        att = 1.0 / dot(att_dist, u_lightAtt[0]);
    }
   
    // Calculate spot attenuation
    if (u_lightSpotCutoff[0] < 180.0)
    {   vec3 S = normalize(v_S_TS); // normalized spot direction
        float spotDot; // Cosine of angle between L and spotdir
        float spotAtt; // Spot attenuation
        spotDot = dot(-L, S);
        if (spotDot < u_lightSpotCosCut[0]) spotAtt = 0.0;
        else spotAtt = max(pow(spotDot, u_lightSpotExp[0]), 0.0);
        att *= spotAtt;
    }
   
    // Compute diffuse lighting
    float diffFactor = max(dot(L,N), 0.0) ;
   
    // Compute specular lighting
    float specFactor = pow(max(dot(N,H), 0.0), u_matShininess);
   
    // Add ambient & diffuse light components
    o_fragColor = u_matEmissive +
                   u_globalAmbient +
                   att * u_lightAmbient[0] * u_matAmbient +
                   att * u_lightDiffuse[0] * u_matDiffuse * diffFactor;
   
    // Componentwise multiply w. texture color
    o_fragColor *= texture(u_matTexture0, v_texCoord);
   
    // Add finally the specular part
    o_fragColor += att * u_lightSpecular[0] * u_matSpecular * specFactor;

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));

    // Apply fog by blending over distance
    if (u_camFogIsOn)
        o_fragColor = fogBlend(v_P_VS, o_fragColor);

    // Apply stereo eye separation
    if (u_camProjection > 1)
        doStereoSeparation();
}
//-----------------------------------------------------------------------------
