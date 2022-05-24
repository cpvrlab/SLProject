//#############################################################################
//  File:      ADSTex.vert
//  Purpose:   GLSL vertex program for ambient, diffuse & specular per vertex 
//             point lighting with texture mapping.
//  Date:      February 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec4    a_position;     // Vertex position attribute
in      vec3    a_normal;       // Vertex normal attribute
in      vec2    a_texCoord;     // Vertex texture coord. attribute

uniform mat4    u_mMatrix;      // Model matrix (object to world transform)
uniform mat4    u_vMatrix;      // View matrix (world to camera transform)
uniform mat4    u_pMatrix;      // Projection matrix (camera to normalize device coords.)

uniform vec4    u_globalAmbi;   // global ambient intensity (Iaglobal)
uniform vec3    u_lightPosVS;   // light position in view space
uniform vec3    u_lightSpotDir; // light direction in view space
uniform vec4    u_lightAmbi;    // light ambient light intensity (Ia)
uniform vec4    u_lightDiff;    // light diffuse light intensity (Id)
uniform vec4    u_lightSpec;    // light specular light intensity (Is)
uniform vec4    u_matAmbi;      // material ambient reflection (ka)
uniform vec4    u_matDiff;      // material diffuse reflection (kd)
uniform vec4    u_matSpec;      // material specular reflection (ks)
uniform vec4    u_matEmis;      // material emissiveness (ke)
uniform float   u_matShin;      // material shininess exponent

out     vec4    v_color;        // The resulting color per vertex
out     vec2    v_texCoord;     // texture coordinate at vertex
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

    // eye position is the inverse of the vertex pos. in VS
    vec3 E = normalize(-P_VS);

    // vector from P_VS to the light in VS
    vec3 L = normalize(u_lightPosVS - P_VS);
   
    // Normalized halfvector between N and L
    vec3 H = normalize(L+E);

    // diffuse factor
    float diffFactor = max(dot(N,L), 0.0);

    // specular factor
    float specFactor = pow(max(dot(N,H), 0.0), u_matShin);

    // Calculate the full Blinn/Phong light equation
    v_color =   u_matEmis +
                u_globalAmbi * u_matAmbi +
                u_lightAmbi * u_matAmbi +
                u_lightDiff * u_matDiff * diffFactor +
                u_lightSpec * u_matSpec * specFactor;

    // Set the texture coord. output for interpolated tex. coords.
    v_texCoord = a_texCoord;

    // Set the transformes vertex position
    gl_Position = u_pMatrix * mvMatrix * a_position;
}
//-----------------------------------------------------------------------------
