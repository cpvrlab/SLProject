//#############################################################################
//  File:      ADSTex.vert
//  Purpose:   GLSL vertex program for ambient, diffuse & specular per vertex 
//             point lighting with texture mapping.
//  Date:      February 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

//-----------------------------------------------------------------------------
attribute   vec4     a_position;       // Vertex position attribute
attribute   vec3     a_normal;         // Vertex normal attribute
attribute   vec2     a_texCoord;       // Vertex texture coord. attribute

uniform     mat4     u_mvMatrix;       // modelView matrix
uniform     mat4     u_mvpMatrix;      // = projection * modelView
uniform     mat3     u_nMatrix;        // normal matrix=transpose(inverse(mv))

uniform     vec4     u_globalAmbi;     // global ambient intensity (Iaglobal)
uniform     vec3     u_lightPosVS;     // light position in view space
uniform     vec3     u_lightDirVS;     // light direction in view space
uniform     vec4     u_lightAmbient;   // light ambient light intensity (Ia)
uniform     vec4     u_lightDiffuse;   // light diffuse light intensity (Id)
uniform     vec4     u_lightSpecular;  // light specular light intensity (Is)
uniform     vec4     u_matAmbient;     // material ambient reflection (ka)
uniform     vec4     u_matDiffuse;     // material diffuse reflection (kd)
uniform     vec4     u_matSpecular;    // material specular reflection (ks)
uniform     vec4     u_matEmissive;    // material emissiveness (ke)
uniform     float    u_matShininess;   // material shininess exponent

varying     vec4     v_color;          // The resulting color per vertex
varying     vec2     v_texCoord;       // texture coordinate at vertex

//-----------------------------------------------------------------------------
void main()
{     
   // transform vertex pos into view space
   vec3 P_VS = vec3(u_mvMatrix * a_position);

   // transform normal into view space
   vec3 N = normalize(u_nMatrix * a_normal);

   // eye position is the inverse of the vertex pos. in VS
   vec3 E = normalize(-P_VS);

   // vector from P_VS to the light in VS
   vec3 L = normalize(u_lightPosVS - P_VS);
   
   // Normalized halfvector between N and L
   vec3 H = normalize(L+E);

   // diffuse factor
   float diffFactor = max(dot(N,L), 0.0);

   // specular factor
   float specFactor = pow(max(dot(N,H), 0.0), u_matShininess);

   // Calculate the full Blinn/Phong light equation 
   v_color =  u_matEmissive +
              u_globalAmbi * u_matAmbient +
              u_lightAmbient * u_matAmbient + 
              u_lightDiffuse * u_matDiffuse * diffFactor +
              u_lightSpecular * u_matSpecular * specFactor;

   // Set the texture coord. varying for interpolated tex. coords.
   v_texCoord = a_texCoord;

   // Set the transformes vertex position           
   gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
