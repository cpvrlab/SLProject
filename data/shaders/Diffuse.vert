//#############################################################################
//  File:      Diffuse.vert
//  Purpose:   GLSL vertex program for simple diffuse per vertex lighting
//  Date:      September 2012 (HS12)
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4 a_position; // Vertex position attribute
layout (location = 1) in vec3 a_normal;   // Vertex normal attribute

uniform mat4     u_mvpMatrix;      // = projection * modelView
uniform mat3     u_nMatrix;        // normal matrix=transpose(inverse(mv))
uniform vec3     u_lightSpotDir; // light direction in view space
uniform vec4     u_lightDiff;   // diffuse light intensity (Id)
uniform vec4     u_matDiff;     // diffuse material reflection (kd)

out     vec4     diffuseColor;     // The resulting color per vertex
//-----------------------------------------------------------------------------
void main()
{  
   // Transform the normal with the normal matrix
    vec3 N = normalize(u_nMatrix * a_normal);
   
   // The diffuse reflection factor is the cosine of the angle between N & L
   float diffFactor = max(dot(N, u_lightSpotDir), 0.0);
   
   // Scale down the diffuse light intensity
   vec4 Id = u_lightDiff * diffFactor;
   
   // The color is light multiplied by material reflection
   diffuseColor = Id * u_matDiff;
   
   // Set the transformes vertex position           
   gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
