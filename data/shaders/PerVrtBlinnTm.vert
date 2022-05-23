//#############################################################################
//  File:      PerVrtBlinnTm.vert
//  Purpose:   GLSL vertex program for per vertex Blinn-Phong lighting
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
layout (location = 0) in vec4  a_position;  // Vertex position attribute
layout (location = 1) in vec3  a_normal;    // Vertex normal attribute
layout (location = 2) in vec2  a_uv0;       // Vertex texture attribute

uniform mat4  u_mMatrix;    // Model matrix (object to world transform)
uniform mat4  u_vMatrix;    // View matrix (world to camera transform)
uniform mat4  u_pMatrix;    // Projection matrix (camera to normalize device coords.)

uniform bool   u_lightIsOn[NUM_LIGHTS];     // flag if light is on
uniform vec4   u_lightPosVS[NUM_LIGHTS];    // position of light in view space
uniform vec4   u_lightAmbi[NUM_LIGHTS];     // ambient light intensity (Ia)
uniform vec4   u_lightDiff[NUM_LIGHTS];     // diffuse light intensity (Id)
uniform vec4   u_lightSpec[NUM_LIGHTS];     // specular light intensity (Is)
uniform vec3   u_lightSpotDir[NUM_LIGHTS];  // spot direction in view space
uniform float  u_lightSpotDeg[NUM_LIGHTS];  // spot cutoff angle 1-180 degrees
uniform float  u_lightSpotCos[NUM_LIGHTS];  // cosine of spot cutoff angle
uniform float  u_lightSpotExp[NUM_LIGHTS];  // spot exponent
uniform vec3   u_lightAtt[NUM_LIGHTS];      // attenuation (const,linear,quadr.)
uniform bool   u_lightDoAtt[NUM_LIGHTS];    // flag if att. must be calc.
uniform vec4   u_globalAmbi;                // Global ambient scene color
uniform float  u_oneOverGamma;              // 1.0f / Gamma correction value

uniform vec4   u_matAmbi;       // ambient color reflection coefficient (ka)
uniform vec4   u_matDiff;       // diffuse color reflection coefficient (kd)
uniform vec4   u_matSpec;       // specular color reflection coefficient (ks)
uniform vec4   u_matEmis;       // emissive color for self-shining materials
uniform float  u_matShin;       // shininess

out     vec3   v_P_VS;          // Point of illumination in view space (VS)
out     vec4   v_color;         // Ambient & diffuse color at vertex
out     vec4   v_specColor;     // Specular color at vertex
out     vec2   v_uv0;           // texture coordinate at vertex
//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces the include pragma by the file
#pragma include "lightingBlinnPhong.glsl"
//-----------------------------------------------------------------------------
void main()
{
    vec4 Ia = vec4(0.0); // Accumulated ambient light intensity at v_P_VS
    vec4 Id = vec4(0.0); // Accumulated diffuse light intensity at v_P_VS
    vec4 Is = vec4(0.0); // Accumulated specular light intensity at v_P_VS

    mat4 mvMatrix = u_vMatrix * u_mMatrix;
    v_P_VS = vec3(mvMatrix * a_position);

    mat3 invMvMatrix = mat3(inverse(mvMatrix));
    mat3 nMatrix = transpose(invMvMatrix);
    vec3 N = vec3(nMatrix * a_normal);

    vec3 E = normalize(-v_P_VS);

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

    // Set the texture coord. output for interpolated tex. coords.
    v_uv0 = a_uv0.xy;
   
    // Sum up all the reflected color components except the specular
    v_color =  u_matEmis +
               u_globalAmbi +
               Ia * u_matAmbi +
               Id * u_matDiff;
   
    // Calculate the specular reflection separately 
    v_specColor =  Is * u_matSpec;

    // For correct alpha blending overwrite alpha component
    v_color.a = u_matDiff.a;

    // Set the transformes vertex position   
    gl_Position = u_pMatrix * mvMatrix * a_position;
}
//-----------------------------------------------------------------------------
