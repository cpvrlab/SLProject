//#############################################################################
//  File:      PerPixBlinnAoSm.vert
//  Purpose:   GLSL vertex shader for per pixel Blinn-Phong lighting with 
//             shadow mapping and ambient occlusion.
//  Date:      July 2019
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;  // Vertex position attribute
layout (location = 1) in vec3  a_normal;    // Vertex normal attribute
layout (location = 3) in vec2  a_uv1;       // Vertex tex.coord. 2 for AO

uniform mat4  u_mMatrix;    // Model matrix
uniform mat4  u_vMatrix;    // View matrix
uniform mat4  u_pMatrix;    // Projection matrix

out     vec3  v_P_VS;       // Point of illumination in view space (VS)
out     vec3  v_P_WS;       // Point of illumination in world space (WS)
out     vec3  v_N_VS;       // Normal at P_VS in view space
out     vec2  v_uv1;        // Texture coordinate 1 output for AO
//-----------------------------------------------------------------------------
void main(void)
{
    v_uv1 = a_uv1;  // pass ambient occlusion tex.coord. 2 for interpolation

    mat4 mvMatrix = u_vMatrix * u_mMatrix;
    v_P_VS = vec3(mvMatrix *  a_position); // vertex position in view space
    v_P_WS = vec3(u_mMatrix * a_position);   // vertex position in world space
    mat3 invMvMatrix = mat3(inverse(mvMatrix));
    mat3 nMatrix = transpose(invMvMatrix);
    v_N_VS = vec3(nMatrix * a_normal);     // vertex normal in view space
    gl_Position = u_pMatrix * mvMatrix * a_position;
}
//-----------------------------------------------------------------------------
