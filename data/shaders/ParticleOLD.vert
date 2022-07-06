//#############################################################################
//  File:      Particle.vert
//  Purpose:   GLSL vertex program for particles drawing
//  Date:      October 2021
//  Authors:   Affolter Marc
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################
precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in  vec3  a_position;       // Particle position attribute
layout (location = 2) in  float  a_startTime;     // Particle start time attribute
layout (location = 4) in  float a_rotation;       // Particle rotation attribute

uniform float u_time;       // Simulation time
uniform float u_tTL;        // Time to live of a particle
uniform vec4 u_pGPosition;  // Particle Generator position

out vertex {
    float transparency; // Transparency of a particle
    float rotation;     // Rotation of a particle
} vert;

uniform mat4 u_vMatrix;    // Modelview matrix

//-----------------------------------------------------------------------------
void main()
{
    float age = u_time - a_startTime;           // Get the age of the particle
    if(age < 0.0){
        vert.transparency = 0.0;                // To be discard, because the particle is to be born
    }
    else{
        vert.transparency = 1.0 - age / u_tTL;  // Get by the ratio age:lifetime
    }
    
    vert.rotation = a_rotation;         

    // Modelview matrix multiplicate with (particle position + particle generator position)
    // Calculate position in view space
    gl_Position =  u_vMatrix * (vec4(a_position, 1) + u_pGPosition);

}
//-----------------------------------------------------------------------------
