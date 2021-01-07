//#############################################################################
//  File:      SkyBox.vert
//  Purpose:   GLSL vertex program for unlit skybox with a cube map
//  Author:    Marcus Hudritsch
//  Date:      October 2017
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;     // Vertex position attribute

uniform mat4    u_mvpMatrix;    // = projection * modelView

out     vec3    v_uv1;          // texture coordinate at vertex
//-----------------------------------------------------------------------------
void main()
{
    v_uv1 = normalize(vec3(a_position));
   
    // Set the transformes vertex position   
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
