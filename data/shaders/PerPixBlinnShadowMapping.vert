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

attribute   vec4  a_position;               //!< Vertex position attribute
attribute   vec3  a_normal;                 //!< Vertex normal attribute
attribute   vec2  a_texCoord;               //!< Vertex texture coordiante attribute

uniform     mat4  u_mMatrix;                //!< model matrix
uniform     mat4  u_mvMatrix;               //!< modelview matrix
uniform     mat3  u_nMatrix;                //!< normal matrix=transpose(inverse(mv))
uniform     mat4  u_mvpMatrix;              //!< = projection * modelView
uniform     mat4  u_lightProjection[8];     //!< projection matrices for lights
uniform     bool  u_lightCreatesShadows[8]; //!< flag if light creates shadows

varying     vec3  v_P_VS;                   //!< Point of illumination in view space (VS)
varying     vec3  v_P_WS;                   //!< Point of illumination in world space (WS)
varying     vec4  v_P_LS[8];                //!< Point of illuminations in object space of the lights
varying     vec3  v_N_VS;                   //!< Normal at P_VS in view space
varying     vec2  v_texCoord;               //!< Texture coordiante varying

//-----------------------------------------------------------------------------
void main(void)
{
    v_P_VS = vec3(u_mvMatrix * a_position);
    v_P_WS = vec3(u_mMatrix * a_position);
    v_N_VS = vec3(u_nMatrix * a_normal);

    for (int i = 0; i < 8; i++)
        if (u_lightCreatesShadows[i])
            v_P_LS[i] = u_lightProjection[i] * u_mMatrix * a_position;

    v_texCoord = a_texCoord;

    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
