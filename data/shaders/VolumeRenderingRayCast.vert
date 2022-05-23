//#############################################################################
//  File:      VolumeRenderingRayCast.vert
//  Purpose:   Base vertex shader that allows for raycast volume rendering through
//             a proxy geometry (usually a cube). The position of each vertex is
//             copied into a output, that provides the entry position of the
//             view ray to the according fragment shader for further calculations.
//  Date:      March 2015
//  Authors:   Manuel Frischknecht, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;     // Vertex position attribute

uniform mat4  u_mMatrix;    // Model matrix (object to world transform)
uniform mat4  u_vMatrix;    // View matrix (world to camera transform)
uniform mat4  u_pMatrix;    // Projection matrix (camera to normalize device coords.)

out     vec3  v_raySource;  //The source coordinate of the view (in model coords)
//-----------------------------------------------------------------------------

void main()
{
   v_raySource = a_position.xyz;
   gl_Position = u_pMatrix * u_vMatrix * u_mMatrix * a_position;
}
//-----------------------------------------------------------------------------
