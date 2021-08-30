//#############################################################################
//  File:      TextureOnly3D.vert
//  Purpose:   GLSL vertex program for 3D texture mapping only
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;     // Vertex position attribute

uniform mat4     u_mvpMatrix;   // = projection * modelView
uniform mat4     u_tMatrix;     // texture transform matrix

out     vec4     v_texCoord3D;  // texture coordinate at vertex
//-----------------------------------------------------------------------------
void main()
{
    // For 3D texturing we use the vertex position as texture coordinate
    // transformed by the texture matrix
    v_texCoord3D = u_tMatrix * a_position;
   
    // Set the transformes vertex position   
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
