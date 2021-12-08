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

//Need to be print out in the correct order
out vec3 tf_position;   // To transform feedback
out vec3 tf_velocity;   // To transform feedback
out float tf_startTime; // To transform feedback
out vec3 tf_initialVelocity; // To transform feedback

out  vec4   v_particleColor;
//-----------------------------------------------------------------------------
void main()
{
    vec4 P = vec4(a_position.xyz, 1.0);
    gl_Position = P; 

    tf_position = a_position;
    tf_velocity = a_velocity;
    tf_startTime = a_startTime;
    tf_initialVelocity = a_initialVelocity;
    if( u_time >= a_startTime ) {
        float age = u_time - a_startTime;
        if( age > u_tTL ) {
            // The particle is past its lifetime, recycle.
            tf_position = vec3(0.0);
            tf_velocity = a_initialVelocity;
            tf_startTime = u_time;
            } else {
            // The particle is alive, update.
            tf_position += tf_velocity * u_deltaTime;
            //tf_velocity += u_deltaTime * u_acceleration;
        }

    }
    v_particleColor = vec4(0,0,0,0);
}
//-----------------------------------------------------------------------------
