//#############################################################################
//  File:      PerPixTmBackground.vert
//  Purpose:   GLSL vertex program for background texture mapping
//  Author:    Marcus Hudritsch
//  Date:      September 2020
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;     // Vertex position attribute

uniform mat4    u_mvpMatrix;    // = projection * modelView
//-----------------------------------------------------------------------------
void main()
{
    // Set the transformes vertex position   
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
