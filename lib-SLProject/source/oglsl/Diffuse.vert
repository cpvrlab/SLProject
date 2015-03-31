//#############################################################################
//  File:      Difuse.vert
//  Purpose:   GLSL vertex program for simple diffuse per vertex lighting
//  Date:      September 2012 (HS12)
//  Copyright: Marcus Hudritsch
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

//-----------------------------------------------------------------------------
attribute   vec4     a_position;       // Vertex position attribute
attribute   vec3     a_normal;         // Vertex normal attribute

uniform     mat4     u_mvpMatrix;      // = projection * modelView
uniform     mat3     u_nMatrix;        // normal matrix=transpose(inverse(mv))
uniform     vec3     u_lightDirVS;     // light direction in view space
uniform     vec4     u_lightDiffuse;   // diffuse light intensity (Id)
uniform     vec4     u_matDiffuse;     // diffuse material reflection (kd)

varying     vec4     v_color;          // The resulting color per vertex

//-----------------------------------------------------------------------------
void main()
{  
   // Transform the normal with the normal matrix
    vec3 N = normalize(u_nMatrix * a_normal);
   
   // The diffuse reflection factor is the cosine of the angle between N & L
   float diffFactor = max(dot(N, u_lightDirVS), 0.0);
   
   // Scale down the diffuse light intensity
   vec4 Id = u_lightDiffuse * diffFactor;
   
   // The color is light multiplied by material reflection
   v_color = Id * u_matDiffuse;
   
   // Set the transformes vertex position           
   gl_Position = u_mvpMatrix * a_position;
}

//-----------------------------------------------------------------------------
