//#############################################################################
//  File:      ADS.vert
//  Purpose:   GLSL vertex program for ambient, diffuse & specular per vertex 
//             point lighting.
//  Date:      September 2012 (HS12)
//  Copyright: Marcus Hudritsch, Switzerland
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

//-----------------------------------------------------------------------------
attribute   vec4     a_position;       // Vertex position attribute
attribute   vec3     a_normal;         // Vertex normal attribute

uniform     mat4     u_mvMatrix;       // modelView matrix
uniform     mat4     u_mvpMatrix;      // = projection * modelView
uniform     mat3     u_nMatrix;        // normal matrix=transpose(inverse(mv))

uniform     vec4     u_globalAmbi;     // global ambient intensity (Iaglobal)
uniform     vec3     u_lightPosVS;     // light position in view space
uniform     vec3     u_lightDirVS;     // light direction in view space
uniform     vec4     u_lightAmbi;      // light ambient light intensity (Ia)
uniform     vec4     u_lightDiff;      // light diffuse light intensity (Id)
uniform     vec4     u_lightSpec;      // light specular light intensity (Is)
uniform     float    u_lightSpotCut;   // lights spot cutoff angle
uniform     float    u_lightSpotCos;   // cosine of the lights spot cutoff angle
uniform     float    u_lightSpotExp;   // lights spot exponent
uniform     vec3     u_lightAtt;       // light attenuation coeff.
uniform     vec4     u_matAmbi;        // material ambient reflection (ka)
uniform     vec4     u_matDiff;        // material diffuse reflection (kd)
uniform     vec4     u_matSpec;        // material specular reflection (ks)
uniform     vec4     u_matEmis;        // material emissiveness (ke)
uniform     float    u_matShine;       // material shininess exponent

varying     vec4     v_color;          // The resulting color per vertex

//-----------------------------------------------------------------------------
void main()
{        
   // transform vertex pos into view space
   vec3 P_VS = vec3(u_mvMatrix * a_position);
   
   // transform normal into view space
   vec3 N = normalize(u_nMatrix * a_normal);
   
   // vector from P_VS to the light in VS
   vec3 L = u_lightPosVS - P_VS;

   // normalize light vector
   L = normalize(L);

   // diffuse factor
   float diffFactor = max(dot(N,L), 0.0);

   // Calculate the full Blinn/Phong light equation 
   v_color =  u_lightDiff * u_matDiff * diffFactor;

   // Set the transformes vertex position           
   gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
