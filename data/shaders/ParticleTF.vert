//#############################################################################
//  File:      ParticleTF.vert
//  Purpose:   GLSL vertex program for particles updating
//  Date:      December 2021
//  Authors:   Affolter Marc
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec3 a_position;       // Particle position attribute
layout (location = 1) in vec3 a_velocity;       // Particle velocity attribute
layout (location = 2) in float a_startTime;     // Particle start time attribute
layout (location = 3) in vec3 a_initialVelocity;// Particle initial velocity attribute
layout (location = 4) in float a_rotation;      // Particle rotation

uniform float u_time;           // Simulation time
uniform float u_deltaTime;      // Elapsed time between frames
uniform vec3 u_acceleration;    // Particle acceleration
uniform float u_tTL;            // Particle lifespan
uniform vec3 u_pGPosition;  // Particle Generator position

//Need to be print out in the correct order
out vec3 tf_position;           // To transform feedback
out vec3 tf_velocity;           // To transform feedback
out float tf_startTime;         // To transform feedback
out vec3 tf_initialVelocity;    // To transform feedback
out float tf_rotation;          // To transform feedback

//-----------------------------------------------------------------------------
void main()
{
    vec4 P = vec4(a_position.xyz, 1.0); // Need to be here for the compilation
    gl_Position = P;                    // Need to be here for the compilation

    tf_position = a_position;   // Init the output variable
    tf_velocity = a_velocity;   // Init the output variable
    tf_startTime = a_startTime; // Init the output variable
    tf_initialVelocity = a_initialVelocity; // Init the output variable
    tf_rotation = a_rotation; // Init the output variable
    //tf_rotation = mod(tf_rotation+((u_tTL/360.0)* u_deltaTime),360.0);
    //tf_rotation = mod(tf_rotation+ 0.05, 360.0);
    tf_rotation = mod(tf_rotation + (0.5*u_deltaTime), 360.0);
    if( u_time >= a_startTime ) {   // Check if the particle is born
        float age = u_time - a_startTime;   // Get the age of the particle
        if( age > u_tTL ) {                 // Check if the particle is dead
            // The particle is past its lifetime, recycle.
            tf_position = u_pGPosition;         // Reset position
            tf_velocity = a_initialVelocity;    // Reset velocity
            tf_startTime = u_time;              // Reset start time to actual time
            } else {
            // The particle is alive, update.
            tf_position += tf_velocity * u_deltaTime;   // Scale the translation by the time
            //tf_velocity += u_deltaTime * u_acceleration;  // Amplify the velocity
        }
    }else{
        tf_position = u_pGPosition;         // Set position (for world space)
    }
}
//-----------------------------------------------------------------------------
