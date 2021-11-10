//#############################################################################
//  File:      ParticleTD.vert
//  Purpose:   GLSL vertex program for simple per vertex attribute color
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in vec3 a_position;
in vec3 a_velocity;
in float a_startTime;
in vec3 a_initialVelocity;

uniform float u_time;  // Simulation time
uniform float u_deltaTime;     // Elapsed time between frames
uniform vec3 u_acceleration;  // Particle acceleration
uniform float u_tTL;  // Particle lifespan

out vec3 td_position;   // To transform feedback
out vec3 td_velocity;   // To transform feedback
out float td_startTime; // To transform feedback

out  vec4   v_particleColor;
//-----------------------------------------------------------------------------
void main()
{
    vec4 P = vec4(a_position.x, a_position.y, 0.0, 1.0);
    gl_Position = P; 
    td_position = vec3(0,0,0);
    td_velocity = vec3(0,0,0);
    td_startTime = 0.0f;

    v_particleColor = vec4(0,0,0,0);
}
//-----------------------------------------------------------------------------
