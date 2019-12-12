//#############################################################################
//  File:      WaveShader.vert
//  Purpose:   GLSL vertex program that builds a sine waved water plane with
//             it correct wave normals for cube mapping in the fragment prog.
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
uniform     float u_t;           // time
uniform     float u_h;           // height of the wave in y direction
uniform     float u_a;           // frequency in x direction
uniform     float u_b;           // frequency in y direction

varying     vec3  v_N_VS;        // Normal in viewspace (VS)
varying     vec3  v_P_VS;        // Vertex in viewspace

//-----------------------------------------------------------------------------
void main(void)
{  
    vec4 p = a_position; // vertex in the x-y-plane
   
    // Calculate z with sine waves shifted by t
    p.z = u_h * sin(u_t + u_a*p.x) * sin(u_t + u_b*p.y);
    v_P_VS = vec3(u_mvMatrix * p);
   
    // Calculate wave normal in view coords 
    float ax = u_a*p.x;
    float by = u_b*p.y;
    float ha = u_h*u_a;
    v_N_VS = vec3(-ha*sin(u_t+by)*cos(u_t+ax), -ha*sin(u_t+ax)*cos(u_t+by), 1.0);
    v_N_VS = vec3(u_nMatrix * v_N_VS);
   
    gl_Position = u_mvpMatrix * p;
}
//-----------------------------------------------------------------------------
