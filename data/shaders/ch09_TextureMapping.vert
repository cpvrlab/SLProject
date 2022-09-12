//#############################################################################
//  File:      ch09_TextureMapping.vert
//  Purpose:   GLSL vertex program for per pixel lighting with texture mapping
//  Date:      February 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec4    a_position;     // Vertex position attribute
in      vec3    a_normal;       // Vertex normal attribute
in      vec2    a_uv;           // Vertex texture coordinate attribute

uniform mat4    u_mMatrix;      // Model matrix (object to world transform)
uniform mat4    u_vMatrix;      // View matrix (world to camera transform)
uniform mat4    u_pMatrix;      // Projection matrix (camera to normalize device coords.)

out     vec3    v_P_VS;         // Point of illumination in view space (VS)
out     vec3    v_N_VS;         // Normal at P_VS in view space
out     vec2    v_uv;           // Texture coordinate at vertex
//-----------------------------------------------------------------------------
void main()
{
    // Transform vertex position into view space
    mat4 mvMatrix = u_vMatrix * u_mMatrix;
    v_P_VS = vec3(mvMatrix * a_position);

    // Transform normal w. transposed, inverse model matrix
    mat3 invMvMatrix = mat3(inverse(mvMatrix));
    mat3 nMatrix = transpose(invMvMatrix);
    v_N_VS = vec3(nMatrix * a_normal);

    // Set the texture coord. output for interpolated tex. coords.
    v_uv = a_uv;

    // Apply model, view and projection matrix
    gl_Position = u_pMatrix * mvMatrix * a_position;
}
//-----------------------------------------------------------------------------
