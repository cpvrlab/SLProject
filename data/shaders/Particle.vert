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
in          vec2     a_position;       // Vertex position attribute
in          vec2     a_texCoord;       // Vertex texture coord. attribute


uniform mat4 u_mvMatrix;  // modelview matrix
uniform vec3 u_offset;	// Object offset

//-----------------------------------------------------------------------------
void main()
{
    vec4 P = vec4(a_position.x, a_position.y, 0.0, 1.0);
    P.xyz += u_offset;
    gl_Position = u_mvMatrix * P; 
}
//-----------------------------------------------------------------------------
