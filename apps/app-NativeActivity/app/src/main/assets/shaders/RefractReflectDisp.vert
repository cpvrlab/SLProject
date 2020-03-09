//#############################################################################
//  File:      RefractReflectDisp.vert
//  Purpose:   GLSL fragment program for refraction- & reflection mapping with
//             chromatic dispersion
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

//-----------------------------------------------------------------------------

attribute vec4 a_position;          // Vertex position attribute
attribute vec3 a_normal;            // Vertex normal attribute

uniform mat4   u_mvMatrix;          // modelview matrix 
uniform mat4   u_mvpMatrix;         // = projection * modelView
uniform mat4   u_invMvMatrix;       // inverse modelview
uniform mat3   u_nMatrix;           // normal matrix=transpose(inverse(mv))

uniform vec4   u_lightPosVS[8];     // position of light in view space
uniform vec4   u_lightSpecular[8];  // specular light intensity (Is)

uniform vec4   u_matAmbient;        // ambient color reflection coefficient (ka)
uniform vec4   u_matDiffuse;        // diffuse color reflection coefficient (kd)
uniform vec4   u_matSpecular;       // specular color reflection coefficient (ks)
uniform vec4   u_matEmissive;       // emissive color for selfshining materials
uniform float  u_matShininess;      // shininess exponent

varying vec3   v_I_VS;              // Incident ray at point of illum. in VS 
varying vec3   v_N_VS;              // normal ray at point of illum. in viewspace
varying vec3   v_R_OS;              // Reflected ray in object space
varying float  v_F_Theta;           // Fresnel reflection coefficient
varying vec4   v_specColor;         // Specular color at vertex

//-----------------------------------------------------------------------------
// Schlick's approximation of the Fresnel reflection coefficient
// theta: angle between normal & incident ray in radians in radians in radians in radians in radians
// F0: reflection coefficient at tetha=0
float F_theta(float theta, float F0)
{
    return F0 + (1.0-F0) * pow(1.0-theta, 5.0);
}

//-----------------------------------------------------------------------------
// Replacement for the GLSL reflect function
vec3 reflect2(vec3 I, vec3 N)
{
    return I - 2.0 * dot(N, I) * N;
}

//-----------------------------------------------------------------------------
void main(void)
{  
    vec3 P_VS = vec3(u_mvMatrix * a_position); // pos. in viewspace
    v_I_VS = normalize(P_VS);                 // incident vector in VS
    v_N_VS = normalize(u_nMatrix * a_normal); // normal vector in VS

    // We have to rotate the relfected & refracted ray by the inverse 
    // modelview matrix back into objekt space. Without that you would see 
    // always the same reflections no matter from where you look
    mat3 iMV = mat3(u_invMvMatrix);
   
    // Calculate reflection vector R and refracted transmission vector T
    v_R_OS = iMV * reflect(v_I_VS, v_N_VS);   // = I - 2.0*dot(N,I)*N;

    // Schlick's approximation of the Fresnel reflection coefficient
    v_F_Theta = F_theta(max(dot(-v_I_VS, v_N_VS), 0.0), 0.2);

    // Specular color for light reflection
    vec3 E = -v_I_VS;                    // eye vector
    vec3 L = u_lightPosVS[0].xyz - P_VS; // Vector from P_VS to the light in VS
    vec3 H = normalize(L+E);            // Normalized halfvector between N and L
    float specFactor = pow(max(dot(v_N_VS,H), 0.0), u_matShininess);
    v_specColor = u_lightSpecular[0] * specFactor * u_matSpecular;

    // Finally transform the vertex position
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
