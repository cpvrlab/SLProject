//#############################################################################
//  File:      ch07_DiffuseLighting.vert
//  Purpose:   GLSL vertex program for simple diffuse per vertex lighting
//  Date:      September 2012 (HS12)
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec4    a_position;     // Vertex position attribute
in      vec3    a_normal;       // Vertex normal attribute

uniform mat4    u_mMatrix;      // Model matrix (object to world transform)
uniform mat4    u_vMatrix;      // View matrix (world to camera transform)
uniform mat4    u_pMatrix;      // Proj. matrix (camera to normalized device coords.)
uniform vec3    u_lightSpotDir; // light direction in view space
uniform vec4    u_lightDiff;    // diffuse light intensity (Id)
uniform vec4    u_matDiff;      // diffuse material reflection (kd)

out     vec4    diffuseColor;   // The resulting color per vertex
//-----------------------------------------------------------------------------
void main()
{
    // Calculate modeview and normal matrix. Do this on GPU and not on CPU
    mat4 mvMatrix = u_vMatrix * u_mMatrix;
    mat3 invMvMatrix = mat3(inverse(mvMatrix));
    mat3 nMatrix = transpose(invMvMatrix);

    // Transform the normal with the normal matrix
    vec3 N = normalize(nMatrix * a_normal);

    // The diffuse reflection factor is the cosine of the angle between N & L
    float diffFactor = max(dot(N, u_lightSpotDir), 0.0);

    // Scale down the diffuse light intensity
    vec4 Id = u_lightDiff * diffFactor;

    // The color is light multiplied by material reflection
    diffuseColor = Id * u_matDiff;

    // Set the transformes vertex position
    gl_Position = u_pMatrix * mvMatrix * a_position;
}
//-----------------------------------------------------------------------------
