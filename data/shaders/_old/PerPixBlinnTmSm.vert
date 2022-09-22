//#############################################################################
//  File:      PerPixBlinnSm.vert
//  Purpose:   GLSL vertex program for per pixel Blinn-Phong lighting with 
//             texture and shadow mapping
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;  // Vertex position attribute
layout (location = 1) in vec3  a_normal;    // Vertex normal attribute
layout (location = 2) in vec2  a_uv0;       // Vertex texture coordinate attribute

uniform mat4  u_mMatrix;    // Model matrix (object to world transform)
uniform mat4  u_vMatrix;    // View matrix (world to camera transform)
uniform mat4  u_pMatrix;    // Projection matrix (camera to normalize device coords.)

out     vec3  v_P_VS;       // Point of illumination in view space (VS)
out     vec3  v_P_WS;       // Point of illumination in world space (WS)
out     vec3  v_N_VS;       // Normal at P_VS in view space
out     vec2  v_uv0;        // Texture coordinate output
//-----------------------------------------------------------------------------
void main(void)
{
    mat4 mvMatrix = u_vMatrix * u_mMatrix;
    mat3 invMvMatrix = mat3(inverse(mvMatrix));
    mat3 nMatrix = transpose(invMvMatrix);

    v_uv0 = a_uv0;  // pass tex. coord. for interpolation

    v_P_VS = vec3(mvMatrix *  a_position);   // vertex position in view space
    v_P_WS = vec3(u_mMatrix * a_position);   // vertex position in world space
    v_N_VS = vec3(nMatrix * a_normal);       // vertex normal in view space

    gl_Position = u_pMatrix * mvMatrix * a_position;
}
//-----------------------------------------------------------------------------