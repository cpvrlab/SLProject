//#############################################################################
//  File:      Geom.vert
//  Purpose:   GLSL vertex shader for geom exercise
//  Date:      November 2021
//  Authors:   Marc Affolter
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec2 aPos;

uniform mat4 u_mvMatrix;  // modelview matrix
uniform vec3 u_offset;	// Object offset

void main()
{
    vec4 P = vec4(aPos.x, aPos.y, 0.0, 1.0);
    P.xyz += u_offset;
    gl_Position = u_mvMatrix * P; 
}