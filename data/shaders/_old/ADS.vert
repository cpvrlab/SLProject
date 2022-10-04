//#############################################################################
//  File:      ADS.vert
//  Purpose:   GLSL vertex program for ambient, diffuse & specular per vertex 
//             point lighting.
//  Date:      September 2012 (HS12)
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec4   a_position;       // Vertex position attribute
in      vec3   a_normal;         // Vertex normal attribute

uniform mat4   u_mMatrix;        // Model matrix (object to world transform)
uniform mat4   u_vMatrix;        // View matrix (world to camera transform)
uniform mat4   u_pMatrix;        // Projection matrix (camera to normalize device coords.)

uniform vec4   u_globalAmbi;     // global ambient intensity (Iaglobal)
uniform vec3   u_lightPosVS;     // light position in view space
uniform vec3   u_lightSpotDir;   // light direction in view space
uniform vec4   u_lightAmbi;      // light ambient light intensity (Ia)
uniform vec4   u_lightDiff;      // light diffuse light intensity (Id)
uniform vec4   u_lightSpec;      // light specular light intensity (Is)
uniform float  u_lightSpotCut;   // lights spot cutoff angle
uniform float  u_lightSpotCos;   // cosine of the lights spot cutoff angle
uniform float  u_lightSpotExp;   // lights spot exponent
uniform vec3   u_lightAtt;       // light attenuation coeff.
uniform vec4   u_matAmbi;        // material ambient reflection (ka)
uniform vec4   u_matDiff;        // material diffuse reflection (kd)
uniform vec4   u_matSpec;        // material specular reflection (ks)
uniform vec4   u_matEmis;        // material emissiveness (ke)
uniform float  u_matShine;       // material shininess exponent

out     vec4   v_color;          // The resulting color per vertex
//-----------------------------------------------------------------------------
void main()
{
   // Calculate modeview and normal matrix. Do this on GPU and not on CPU
   mat4 mvMatrix = u_vMatrix * u_mMatrix;
   mat3 invMvMatrix = mat3(inverse(mvMatrix));
   mat3 nMatrix = transpose(invMvMatrix);

   // transform vertex pos into view space
   vec3 P_VS = vec3(mvMatrix * a_position);
   
   // transform normal into view space
   vec3 N = normalize(nMatrix * a_normal);
   
   // vector from P_VS to the light in VS
   vec3 L = u_lightPosVS - P_VS;

   // normalize light vector
   L = normalize(L);

   // diffuse factor
   float diffFactor = max(dot(N,L), 0.0);

   // Calculate the full Blinn/Phong light equation 
   v_color =  u_lightDiff * u_matDiff * diffFactor;

   // Set the transformes vertex position
   gl_Position = u_pMatrix * mvMatrix * a_position;
}
//-----------------------------------------------------------------------------
