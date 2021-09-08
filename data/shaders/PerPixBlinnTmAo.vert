//#############################################################################
//  File:      PerPixBlinnTm.vert
//  Purpose:   GLSL vertex program for per pixel Blinn lighting w. tex. & ambient occlusion
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;  // Vertex position attribute
layout (location = 1) in vec3  a_normal;    // Vertex normal attribute
layout (location = 2) in vec2  a_uv1;       // Vertex tex.coord. 1 for diffuse color
layout (location = 3) in vec2  a_uv2;       // Vertex tex.coord. 2 for AO

uniform mat4  u_mvMatrix;   // modelview matrix
uniform mat3  u_nMatrix;    // normal matrix=transpose(inverse(mv))
uniform mat4  u_mvpMatrix;  // = projection * modelView

out     vec3  v_P_VS;       // Point of illumination in view space (VS)
out     vec3  v_N_VS;       // Normal at P_VS in view space
out     vec2  v_uv1;        // Texture coordinate output
//-----------------------------------------------------------------------------
void main(void)
{  
    v_uv1 = a_uv1;  // pass diffuse color tex.coord. 1 for interpolation
    v_uv2 = a_uv2;  // pass ambient occlusion tex.coord. 2 for interpolation

    v_P_VS = vec3(u_mvMatrix * a_position);
    v_N_VS = vec3(u_nMatrix * a_normal);
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
