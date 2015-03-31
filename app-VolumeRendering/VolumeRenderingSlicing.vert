//#############################################################################
//  File:      VolumeRenderingSlicing.vert
//  Purpose:   Vertex shader for slice-based volume rendering. The slices are
//             created as multiple quads aligned in a cubic shape.
//  Author:    Manuel Frischknecht
//  Date:      March 2015
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

//-----------------------------------------------------------------------------
attribute   vec4       a_position;             // Vertex position attribute

uniform     vec3       u_voxelScale;           // Voxel scaling coefficients
uniform     mat4       u_mvpMatrix;			   // = projection * modelView
uniform     mat4       u_volumeRotationMatrix; // Rotation matrix for the volume itself
uniform     vec3       u_textureSize;          // size of 3D texture

uniform     sampler3D  u_volume;		       // A 3D texture defining the volume

varying     vec4       v_texCoord;             // Texture coordinate at each vertex

//-----------------------------------------------------------------------------

void main()
{
   //Determine the size of the volume texture (after scaling)
   vec3 size = u_textureSize*u_voxelScale;
   float max_length = max(max(size.x,size.y),size.z);

   //Determine the texture coordinates for each vertex
   //and apply the specified volume rotation
   v_texCoord = u_volumeRotationMatrix * a_position;
   v_texCoord.xyz = 0.5*v_texCoord.xyz+vec3(0.5);
   v_texCoord.xyz *= vec3(max_length)/size;

   // Set the transformed vertex position
   gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
