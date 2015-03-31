//#############################################################################
//  File:      PerPixBlinn.vert
//  Purpose:   GLSL vertex program for per fragment Blinn-Phong lighting
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

attribute   vec4  a_position;    // Vertex position attribute
attribute   vec3  a_normal;      // Vertex normal attribute

uniform     mat4  u_mvMatrix;    // modelview matrix 
uniform     mat3  u_nMatrix;     // normal matrix=transpose(inverse(mv))
uniform     mat4  u_mvpMatrix;   // = projection * modelView

varying     vec3  v_P_VS;        // Point of illumination in view space (VS)
varying     vec3  v_N_VS;        // Normal at P_VS in view space

//-----------------------------------------------------------------------------
void main(void)
{  
    v_P_VS = vec3(u_mvMatrix * a_position);
    v_N_VS = vec3(u_nMatrix * a_normal);  
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
