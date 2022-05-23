//#############################################################################
//  File:      WaveShader.vert
//  Purpose:   GLSL vertex program that builds a sine waved water plane with
//             it correct wave normals for cube mapping in the fragment prog.
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;     // Vertex position attribute
layout (location = 1) in vec3  a_normal;       // Vertex normal attribute

uniform mat4  u_mMatrix;    // Model matrix (object to world transform)
uniform mat4  u_vMatrix;    // View matrix (world to camera transform)
uniform mat4  u_pMatrix;    // Projection matrix (camera to normalize device coords.)

uniform float u_t;           // time
uniform float u_h;           // height of the wave in y direction
uniform float u_a;           // frequency in x direction
uniform float u_b;           // frequency in y direction

out     vec3  v_N_VS;        // Normal in viewspace (VS)
out     vec3  v_P_VS;        // Vertex in viewspace
//-----------------------------------------------------------------------------
void main(void)
{  
    vec4 p = a_position; // vertex in the x-y-plane
   
    // Calculate z with sine waves shifted by t
    p.z = u_h * sin(u_t + u_a*p.x) * sin(u_t + u_b*p.y);

    mat4 mvMatrix = u_vMatrix * u_mMatrix;
    v_P_VS = vec3(mvMatrix * p);
   
    // Calculate wave normal in view coords 
    float ax = u_a*p.x;
    float by = u_b*p.y;
    float ha = u_h*u_a;
    v_N_VS = vec3(-ha*sin(u_t+by)*cos(u_t+ax), -ha*sin(u_t+ax)*cos(u_t+by), 1.0);

    mat3 invMvMatrix = mat3(inverse(mvMatrix));
    mat3 nMatrix = transpose(invMvMatrix);
    v_N_VS = vec3(nMatrix * v_N_VS);
   
    gl_Position = u_pMatrix * mvMatrix * p;
}
//-----------------------------------------------------------------------------
