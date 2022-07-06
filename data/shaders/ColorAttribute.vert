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
layout (location = 0) in vec4 a_position;        // Vertex position attribute
layout (location = 4) in vec4 a_color;           // Vertex color attribute

uniform mat4  u_mMatrix;    // Model matrix (object to world transform)
uniform mat4  u_vMatrix;    // View matrix (world to camera transform)
uniform mat4  u_pMatrix;    // Projection matrix (camera to normalized device coords.)

out     vec4  v_color;      // Resulting color per vertex
//-----------------------------------------------------------------------------
void main(void)
{    
    v_color = a_color;      // Pass the color attribute to the output color

    // Multiply the model-view-projection matrix that transforms then vertex
    // from the local into the world and then into the camera and the into the
    // normalized devive space.
    gl_Position = u_pMatrix * u_vMatrix * u_mMatrix * a_position;
}
//-----------------------------------------------------------------------------
