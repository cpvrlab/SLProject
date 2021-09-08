//#############################################################################
//  File:      WaveShader.frag
//  Purpose:   GLSL fragment program that illuminates the wave from the vertex
//             program with 50% from a cube texture and 50% from a pointlight.
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces #Lights by SLVLights.size()
#pragma define NUM_LIGHTS #Lights
//-----------------------------------------------------------------------------
in      vec3    v_P_VS;              // Point of illumination in viewspace (VS)
in      vec3    v_N_VS;              // Unnormalized normal at P

uniform bool    u_lightIsOn[NUM_LIGHTS];        // flag if light is on
uniform vec4    u_lightPosVS[NUM_LIGHTS];       // position of light in view space
uniform vec4    u_lightAmbi[NUM_LIGHTS];     // ambient light intensity (Ia)
uniform vec4    u_lightDiff[NUM_LIGHTS];     // diffuse light intensity (Id)
uniform vec4    u_lightSpec[NUM_LIGHTS];    // specular light intensity (Is)
uniform vec3    u_lightSpotDir[NUM_LIGHTS];   // spot direction in view space
uniform float   u_lightSpotDeg[NUM_LIGHTS];  // spot cutoff angle 1-180 degrees
uniform float   u_lightSpotCos[NUM_LIGHTS];  // cosine of spot cutoff angle
uniform float   u_lightSpotExp[NUM_LIGHTS];     // spot exponent
uniform vec3    u_lightAtt[NUM_LIGHTS];         // attenuation (const,linear,quadr.)
uniform bool    u_lightDoAtt[NUM_LIGHTS];       // flag if att. must be calc.
uniform vec4    u_globalAmbi;                // Global ambient scene color
uniform float   u_oneOverGamma;                 // 1.0f / Gamma correction value

uniform vec4    u_matAmbi;        // ambient color reflection coefficient (ka)
uniform vec4    u_matDiff;        // diffuse color reflection coefficient (kd)
uniform vec4    u_matSpec;       // specular color reflection coefficient (ks)
uniform vec4    u_matEmis;       // emissive color for self-shining materials
uniform float   u_matShin;      // shininess exponent

out     vec4    o_fragColor;         // output fragment color
//-----------------------------------------------------------------------------
void main()
{
    float nDotL;                  // N dot L = diffuse reflection factor
    float shine;                  // specular reflection factor
   
    vec3 N = normalize(v_N_VS);   // A input normal has not anymore unit length
    vec3 E = normalize(-v_P_VS);  // Vector from p to the eye
    vec3 L = normalize(u_lightPosVS[0].xyz - v_P_VS); // Vector to light
    vec3 H = normalize(L + E);    // Halfvector between L & E    
   
    // Calculate diffuse & specular factors
    nDotL = max(dot(N,L), 0.0);
    if (nDotL==0.0) shine = 0.0; 
    else shine = pow(max(dot(N,H), 0.0), u_matShin);
   
    // Accumulate pointlight reflection
    vec4 matCol =  u_globalAmbi +
                   u_lightAmbi[0] * u_matAmbi +
                   u_lightDiff[0]  * nDotL * u_matDiff +
                   u_lightSpec[0] * shine * u_matSpec;
   
    // Mix the final color fifty-fifty
    o_fragColor = matCol;

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
