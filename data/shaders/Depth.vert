//#############################################################################
//  File:      Depth.vert
//  Purpose:   Simple depth shader
//  Author:    Michael Schertenleib, Marcus Hudritsch
//  Date:      March 2020
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

attribute   vec4  a_position;    // Vertex position attribute

uniform     mat4  u_mvpMatrix;   // = projection * modelView

//-----------------------------------------------------------------------------
void main(void)
{
    gl_Position = u_mvpMatrix * a_position;
}