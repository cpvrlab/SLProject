//#############################################################################
//  File:      ColorAttribute.vert
//  Purpose:   GLSL vertex program for simple per vertex attribute color
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in  vec3  a_position;       // Vertex position attribute
layout (location = 2) in  float  a_startTime;

uniform float u_time;  // simulation time
uniform float u_tTL;  // time to live

out vertex {
    float transparency;
} vert;

uniform mat4 u_mvMatrix;  // modelview matrix

//-----------------------------------------------------------------------------
void main()
{
    float age = u_time - a_startTime;
    vert.transparency = 1.0 - age / u_tTL;
    /*vec4 d = vec4(0,0,0,1.0);
    vec4 P = vec4(a_position.xyz,0.0);
    gl_Position = (u_mvMatrix * d)+P; */

    vec4 P = vec4(a_position.xyz,1.0);
    gl_Position = u_mvMatrix * P; 
}
//-----------------------------------------------------------------------------
