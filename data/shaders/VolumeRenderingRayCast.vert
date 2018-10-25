//#############################################################################
//  File:      VolumeRenderingRayCast.vert
//  Purpose:   Base vertex shader that allows for raycast volume rendering through
//             a proxy geometry (usually a cube). The position of each vertex is
//             copied into a varying, that provides the entry position of the
//             view ray to the according fragment shader for further calculations.
//  Author:    Manuel Frischknecht
//  Date:      March 2015
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

//-----------------------------------------------------------------------------
attribute   vec4     a_position;          // Vertex position attribute

uniform     mat4     u_mvpMatrix;         // = projection * modelView

varying     vec3     v_raySource;         //The source coordinate of the view ray
                                          //(in model coordinates)

//-----------------------------------------------------------------------------

void main()
{
   v_raySource = a_position.xyz; 
   gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
