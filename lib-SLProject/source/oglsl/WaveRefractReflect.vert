//#############################################################################
//  File:      WaveReflectRefract.vert
//  Purpose:   GLSL vertex program that builds a sine waved water plane with
//             it correct wave normals for cube mapping in the fragment prog.
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

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
uniform float  u_matShininess;      // shininess exponent

uniform float  u_t;                 // time
uniform float  u_h;                 // height of the wave in y direction
uniform float  u_a;                 // frequency in x direction
uniform float  u_b;                 // frequency in y direction

varying vec3   v_R_OS;              // Reflected ray in object space
varying vec3   v_T_OS;              // Refracted ray in object space
varying float  v_F_Theta;           // Fresnel reflection coefficient
varying vec4   v_specColor;         // Specular color at vertex

//-----------------------------------------------------------------------------
// Schlick's approximation of the Fresnel reflection coefficient
// theta: angle between normal & incident ray in radians in radians
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
// Replacement for the GLSL refract function
vec3 refract2(vec3 I, vec3 N, float eta)
{
    float NdotI = dot(N,I);
    float k = 1.0 - eta * eta * (1.0 -  NdotI*NdotI);
    if (k < 0.0) return vec3(0);
    else return eta * I - (eta * NdotI + sqrt(k)) * N;
}

//-----------------------------------------------------------------------------
void main(void)
{  
    vec4 p = a_position; // vertex in the x-y-plane
   
    // Calculate z with sine waves shifted by t
    p.z = u_h * sin(u_t + u_a*p.x) * sin(u_t + u_b*p.y);
    vec3 P_VS = vec3(u_mvMatrix * p);
   
    // Incident vector in view spcae
    vec3 I_VS = normalize(P_VS);
   
    // Calculate wave normal in view coords 
    float ax = u_a*p.x;
    float by = u_b*p.y;
    float ha = u_h*u_a;
    vec3 N_VS = vec3(-ha*sin(u_t+by)*cos(u_t+ax), -ha*sin(u_t+ax)*cos(u_t+by), 1.0);
    N_VS = normalize(vec3(u_nMatrix * N_VS));

    // We have to rotate the relfected & refracted ray by the inverse 
    // modelview matrix back into objekt space. Without that you would see 
    // always the same reflections no matter from where you look
    mat3 iMV = mat3(u_invMvMatrix[0].xyz,
                    u_invMvMatrix[1].xyz,
                    u_invMvMatrix[2].xyz);
   
    // Calculate reflection vector R and refracted transmission vector T
    v_R_OS = iMV * reflect(I_VS, N_VS);          // = I - 2.0*dot(N,I)*N;
    v_T_OS = iMV * refract(I_VS, N_VS, 0.66666); // eta = etaAir/etaGlass = 1/1.5

    // Schlick's approximation of the Fresnel reflection coefficient
    v_F_Theta = F_theta(max(dot(-I_VS, N_VS),0.0), 0.1);

    // Specular color for light reflection
    vec3 E = -I_VS;                      // eye vector
    vec3 L = u_lightPosVS[0].xyz - P_VS; // Vector from P_VS to the light in VS
    vec3 H = normalize(L+E);             // Normalized halfvector between N and L
    float specFactor = pow(max(dot(N_VS,H), 0.0), u_matShininess);
    v_specColor = u_lightSpecular[0] * specFactor * u_matSpecular;
   
    // Finally transform the vertex position
    gl_Position = u_mvpMatrix * p;
}
//-----------------------------------------------------------------------------
