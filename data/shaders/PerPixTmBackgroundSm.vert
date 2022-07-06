//#############################################################################
//  File:      PerPixTmBackground.vert
//  Purpose:   GLSL vertex shader for background texture mapping with
//             shadow mapping
//  Date:      November 2020
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;  // Vertex position attribute
layout (location = 1) in vec3  a_normal;    // Vertex normal attribute

uniform mat4  u_mMatrix;    // Model matrix (object to world transform)
uniform mat4  u_vMatrix;    // View matrix (world to camera transform)
uniform mat4  u_pMatrix;    // Projection matrix (camera to normalize device coords.)

out     vec3  v_P_VS;       // Point of illumination in view space (VS)
out     vec3  v_P_WS;       // Point of illumination in world space (WS)
out     vec3  v_N_VS;       // Normal at P_VS in view space
//-----------------------------------------------------------------------------
void main()
{
    mat4 mvMatrix = u_vMatrix * u_mMatrix;
    v_P_VS = vec3(mvMatrix * a_position);
    v_P_WS = vec3(u_mMatrix * a_position);

    mat3 invMvMatrix = mat3(inverse(mvMatrix));
    mat3 nMatrix = transpose(invMvMatrix);
    v_N_VS = vec3(nMatrix * a_normal);

    // Set the transformes vertex position   
    gl_Position = u_pMatrix * mvMatrix * a_position;
}
//-----------------------------------------------------------------------------
