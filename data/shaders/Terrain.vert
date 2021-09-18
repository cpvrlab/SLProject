//#############################################################################
//  File:      Terrain.vert
//  Purpose:   GLSL vertex program for simple diffuse lighting with texture
//  Authors:   Marcus Hudritsch
//  Date:      September 2012 (HS12)
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;  // Vertex position attribute
layout (location = 1) in vec3  a_normal;    // Vertex normal attribute
layout (location = 2) in vec2  a_uv0;       // Vertex texture attribute

uniform mat4   u_mvMatrix;      // modelview matrix
uniform mat3   u_nMatrix;       // normal matrix=transpose(inverse(mv))
uniform mat4   u_mvpMatrix;     // = projection * modelView

uniform bool   u_lightIsOn[8];  // flag if light is on
uniform vec4   u_lightPosVS[8]; // position of light in view space
uniform vec4   u_lightAmbi[8];  // ambient light intensity (Ia)
uniform vec4   u_lightDiff[8];  // diffuse light intensity (Id)
uniform vec4   u_globalAmbi;    // Global ambient scene color

uniform vec4   u_matAmbi;       // ambient color reflection coefficient (ka)
uniform vec4   u_matDiff;       // diffuse color reflection coefficient (kd)

out     vec4   v_color;         // Ambient & diffuse color at vertex
out     vec2   v_uv0;           // texture coordinate at vertex
//-----------------------------------------------------------------------------
void main()
{     
   vec3 P_VS = vec3(u_mvMatrix * a_position);
   vec3 N = normalize(u_nMatrix * a_normal);
   vec3 L = normalize(u_lightPosVS[0].xyz - P_VS);
   
   // Calculate diffuse & specular factors
   float diffFactor = max(dot(N,L), 0.0);
   
   // Set the texture coord. output for interpolated tex. coords.
   v_uv0 = a_uv0.xy;
   
   // Sum up all the reflected color components except the specular
   v_color =  u_globalAmbi +
              u_lightAmbi[0] * u_matAmbi +
              u_lightDiff[0] * u_matDiff * diffFactor;

   // Set the transformes vertex position   
   gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
