//#############################################################################
//  File:      PBR_LightingTm.vert
//  Purpose:   GLSL vertex shader for Cook-Torrance physical based rendering
//             including diffuse irradiance and specular IBL. Based on the
//             physically based rendering (PBR) tutorial with GLSL by Joey de
//             Vries on https://learnopengl.com/#!PBR/Theory
//  Author:    Carlos Arauz, 
//             adapted from PerPixCookTorrance.vert by Marcus Hudritsch
//  Date:      April 2018
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position; // Vertex position attribute
layout (location = 1) in vec3  a_normal;   // Vertex normal attribute
layout (location = 2) in vec2  a_uv1;      // Vertex texture coordinate attribute

uniform     mat4  u_mvMatrix;   // modelview matrix 
uniform     mat3  u_nMatrix;    // normal matrix=transpose(inverse(mv))
uniform     mat4  u_mvpMatrix;  // = projection * modelView
uniform     mat4  u_invMvMatrix;// inverse modelview matrix

out         vec3  v_P_VS;       // Point of illumination in view space (VS)
out         vec3  v_N_VS;       // Normal at P_VS in view space
out         vec3  v_R_OS;       // Reflected ray in object space
out         vec2  v_uv1;        // Texture coordinate output
//-----------------------------------------------------------------------------
void main()
{
    v_P_VS = vec3(u_mvMatrix * a_position);
    v_N_VS = vec3(u_nMatrix * a_normal); 
    v_uv1  = a_uv1;
  
    // Calculate reflection vector R
    vec3 I = normalize(v_P_VS);
    vec3 N = normalize(v_N_VS);
    v_R_OS =  mat3(u_invMvMatrix) * reflect(I, v_N_VS); // = I - 2.0*dot(N,I)*N;
  
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
