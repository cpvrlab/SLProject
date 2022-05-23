//#############################################################################
//  File:      PerPixTmBackground.vert
//  Purpose:   GLSL vertex program for background texture mapping
//  Date:      September 2020
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;     // Vertex position attribute

uniform mat4  u_mMatrix;    // Model matrix (object to world transform)
uniform mat4  u_vMatrix;    // View matrix (world to camera transform)
uniform mat4  u_pMatrix;    // Projection matrix (camera to normalize device coords.)
//-----------------------------------------------------------------------------
void main()
{
    // Set the transformes vertex position
    mat4 mvMatrix = u_vMatrix * u_mMatrix;
    gl_Position = u_pMatrix * mvMatrix * a_position;
}
//-----------------------------------------------------------------------------
