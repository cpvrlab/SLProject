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
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_velocity;
layout (location = 2) in float a_startTime;
layout (location = 3) in vec3 a_initialVelocity;

uniform float u_time;  // Simulation time
uniform float u_deltaTime;     // Elapsed time between frames
uniform vec3 u_acceleration;  // Particle acceleration
uniform vec3 u_offset;  // Particle offset
uniform float u_tTL;  // Particle lifespan

out vec3 td_position;   // To transform feedback
out vec3 td_velocity;   // To transform feedback
out float td_startTime; // To transform feedback

out  vec4   v_particleColor;
//-----------------------------------------------------------------------------
void main()
{
    vec4 P = vec4(a_position.xyz, 1.0);
    gl_Position = P; 

    td_position = a_position;
    td_velocity = a_velocity;
    td_startTime = a_startTime;
    if( u_time >= a_startTime ) {
        float age = u_time - a_startTime;
        if( age > u_tTL ) {
            // The particle is past its lifetime, recycle.
            td_position = u_offset;
            td_velocity = a_initialVelocity;
            td_startTime = u_time;
            } else {
            // The particle is alive, update.
            td_position += td_velocity * u_deltaTime;
            td_startTime += u_acceleration * u_deltaTime;
        }

    }
    v_particleColor = vec4(0,0,0,0);
}
//-----------------------------------------------------------------------------
