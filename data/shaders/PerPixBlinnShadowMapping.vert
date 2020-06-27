//#############################################################################
//  File:      PerPixBlinnShadowMapping.vert
//  Purpose:   GLSL vertex program for per fragment Blinn-Phong lighting
//             (and Shadow mapping)
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

//-----------------------------------------------------------------------------
in      vec4  a_position;               // Vertex position attribute
in      vec3  a_normal;                 // Vertex normal attribute
in      vec2  a_texCoord;               // Vertex texture coordiante attribute

uniform mat4  u_mMatrix;                // model matrix
uniform mat4  u_mvMatrix;               // modelview matrix
uniform mat3  u_nMatrix;                // normal matrix=transpose(inverse(mv))
uniform mat4  u_mvpMatrix;              // = projection * modelView

out     vec3  v_P_VS;                   // Point of illumination in view space (VS)
out     vec3  v_P_WS;                   // Point of illumination in world space (WS)
out     vec3  v_N_VS;                   // Normal at P_VS in view space
out     vec2  v_texCoord;               // Texture coordiante output
//-----------------------------------------------------------------------------
void main(void)
{
    v_P_VS = vec3(u_mvMatrix * a_position);
    v_P_WS = vec3(u_mMatrix * a_position);
    v_N_VS = vec3(u_nMatrix * a_normal);

    v_texCoord = a_texCoord;

    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
