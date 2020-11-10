//#############################################################################
//  File:      PerVrtBlinnColorAttrib.vert
//  Purpose:   GLSL vertex program for per vertex Blinn-Phong lighting with 
//             diffuse color vertex attribute
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces #Lights by SLVLights.size()
#pragma define NUM_LIGHTS #Lights
//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;     // Vertex position attribute
layout (location = 1) in vec3  a_normal;       // Vertex normal attribute
layout (location = 4) in vec4  a_color;        // Vertex color attribute

uniform mat4   u_mvMatrix;          // modelview matrix 
uniform mat3   u_nMatrix;           // normal matrix=transpose(inverse(mv))
uniform mat4   u_mvpMatrix;         // = projection * modelView

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

uniform vec4   u_matAmbi;           // ambient color reflection coefficient (ka)
uniform vec4   u_matSpec;           // specular color reflection coefficient (ks)
uniform vec4   u_matEmis;           // emissive color for self-shining materials
uniform float  u_matShin;           // shininess exponent

out     vec4   v_color;             // The resulting color per vertex
out     vec3   v_P_VS;              // Point of illumination in view space (VS)
//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces the include pragma by the file
#pragma include "lightingBlinnPhong.glsl"
//-----------------------------------------------------------------------------
void main()
{
    vec4 Ia = vec4(0.0); // Accumulated ambient light intensity at v_P_VS
    vec4 Id = vec4(0.0); // Accumulated diffuse light intensity at v_P_VS
    vec4 Is = vec4(0.0); // Accumulated specular light intensity at v_P_VS

    v_P_VS = vec3(u_mvMatrix * a_position);
    vec3 N = normalize(u_nMatrix * a_normal);
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

    // Sum up all the reflected color components
    v_color =  u_matEmis +
               u_globalAmbi +
               Ia * a_color +
               Id * a_color +
               Is * u_matSpec;

    // For correct alpha blending overwrite alpha component
    v_color.a = a_color.a;

    // Apply gamma correction
    v_color.rgb = pow(v_color.rgb, vec3(u_oneOverGamma));

    // Set the transformes vertex position           
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
