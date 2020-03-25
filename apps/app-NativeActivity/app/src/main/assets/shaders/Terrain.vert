//#############################################################################
//  File:      Terrain.vert
//  Purpose:   GLSL vertex program for simple diffuse lighting with texture
//  Author:    Marcus Hudritsch
//  Date:      September 2012 (HS12)
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

//-----------------------------------------------------------------------------
attribute vec4 a_position;          // Vertex position attribute
attribute vec3 a_normal;            // Vertex normal attribute
attribute vec2 a_texCoord;          // Vertex texture coord. attribute

uniform mat4   u_mvMatrix;          // modelview matrix 
uniform mat3   u_nMatrix;           // normal matrix=transpose(inverse(mv))
uniform mat4   u_mvpMatrix;         // = projection * modelView

uniform bool   u_lightIsOn[8];      // flag if light is on
uniform vec4   u_lightPosVS[8];     // position of light in view space
uniform vec4   u_lightAmbient[8];   // ambient light intensity (Ia)
uniform vec4   u_lightDiffuse[8];   // diffuse light intensity (Id)
uniform vec4   u_globalAmbient;     // Global ambient scene color

uniform vec4   u_matAmbient;        // ambient color reflection coefficient (ka)
uniform vec4   u_matDiffuse;        // diffuse color reflection coefficient (kd)

varying vec4   v_color;             // Ambient & diffuse color at vertex
varying vec2   v_texCoord;          // texture coordinate at vertex

//-----------------------------------------------------------------------------
void main()
{     
   vec3 P_VS = vec3(u_mvMatrix * a_position);
   vec3 N = normalize(u_nMatrix * a_normal);
   vec3 L = normalize(u_lightPosVS[0].xyz - P_VS);
   
   // Calculate diffuse & specular factors
   float diffFactor = max(dot(N,L), 0.0);
   
   // Set the texture coord. varying for interpolated tex. coords.
   v_texCoord = a_texCoord.xy;
   
   // Sum up all the reflected color components except the specular
   v_color =  u_globalAmbient +
              u_lightAmbient[0] * u_matAmbient +
              u_lightDiffuse[0] * u_matDiffuse * diffFactor;

   // Set the transformes vertex position   
   gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
