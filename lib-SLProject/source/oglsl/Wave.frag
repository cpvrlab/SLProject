//#############################################################################
//  File:      WaveShader.frag
//  Purpose:   GLSL fragment program that illuminates the wave from the vertex
//             program with 50% from a cube texture and 50% from a pointlight.
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
varying vec3   v_P_VS;              // Point of illumination in viewspace (VS)
varying vec3   v_N_VS;              // Unnormalized normal at P

uniform int    u_numLightsUsed;     // NO. of lights used light arrays
uniform bool   u_lightIsOn[8];      // flag if light is on
uniform vec4   u_lightPosVS[8];     // position of light in view space
uniform vec4   u_lightAmbient[8];   // ambient light intensity (Ia)
uniform vec4   u_lightDiffuse[8];   // diffuse light intensity (Id)
uniform vec4   u_lightSpecular[8];  // specular light intensity (Is)
uniform vec3   u_lightDirVS[8];     // spot direction in view space
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

//-----------------------------------------------------------------------------
void main()
{
    float nDotL;                  // N dot L = diffuse reflection factor
    float shine;                  // specular reflection factor
   
    vec3 N = normalize(v_N_VS);   // A varying normal has not anymore unit length
    vec3 E = normalize(-v_P_VS);  // Vector from p to the eye
    vec3 L = normalize(u_lightPosVS[0].xyz - v_P_VS); // Vector to light
    vec3 H = normalize(L + E);    // Halfvector between L & E    
   
    // Calculate diffuse & specular factors
    nDotL = max(dot(N,L), 0.0);
    if (nDotL==0.0) shine = 0.0; 
    else shine = pow(max(dot(N,H), 0.0), u_matShininess); 
   
    // Accumulate pointlight reflection
    vec4 matCol =  u_globalAmbient +
                   u_lightAmbient[0] * u_matAmbient +
                   u_lightDiffuse[0]  * nDotL * u_matDiffuse + 
                   u_lightSpecular[0] * shine * u_matSpecular;
   
    // Mix the final color fifty-fifty
    gl_FragColor = matCol;
}
//-----------------------------------------------------------------------------