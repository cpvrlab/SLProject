//#############################################################################
//  File:      RefractReflectDisp.frag
//  Purpose:   GLSL fragment program for refraction- & reflection mapping with
//             chromatic dispersion
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
uniform samplerCube  u_texture0;    // Cubic environment texture map

uniform mat4   u_mvMatrix;          // modelview matrix
uniform mat4   u_invMvMatrix;       // inverse modelview 
uniform mat3   u_nMatrix;           // normal matrix=transpose(inverse(mv))

varying vec3   v_I_VS;              // Incident ray at point of illum. in viewspace 
varying vec3   v_N_VS;              // normal ray at point of illum. in viewspace 
varying vec3   v_R_OS;              // Reflected ray in object space
varying floa   v_F_Theta;           // Fresnel reflection coefficient
varying vec4   v_specColor;         // Specular color at vertex


//-----------------------------------------------------------------------------
// Replacement for the GLSL refract function
vec3 refract2(vec3 I, vec3 N, float eta)
{
    float NdotI = dot(N,I);
    float k = 1.0 - eta * eta * (1.0 -  NdotI*NdotI);
    if (k < 0.0) return vec3(0);
    else return eta * I - (eta * NdotI + sqrt(k)) * N;
}
//-----------------------------------------------------------------------------
void main()
{
    // We have to rotate the relfected & refracted ray by the inverse 
    // modelview matrix back into objekt space. Without that you would see 
    // always the same reflections no matter from where you look
    mat3 iMV = mat3(u_invMvMatrix);
   
    // get the reflection & refraction color out of the cubic map
    vec4 reflCol = textureCube(u_texture0, v_R_OS);
   
    //Chromatic dispersion refract rays depending on their wave length
    vec4 refrCol;
    refrCol.r = textureCube(u_texture0, iMV * refract(v_I_VS, v_N_VS, 1.0/1.45)).r;
    refrCol.g = textureCube(u_texture0, iMV * refract(v_I_VS, v_N_VS, 1.0/1.50)).g;
    refrCol.b = textureCube(u_texture0, iMV * refract(v_I_VS, v_N_VS, 1.0/1.55)).b;
    refrCol.a = 1.0;
   
    // Mix the final color with the fast frenel factor
    gl_FragColor = mix(refrCol, reflCol, v_F_Theta);

    // Add specular gloss
    gl_FragColor.rgb += v_specColor.rgb;
}