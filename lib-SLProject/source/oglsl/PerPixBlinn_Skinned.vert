//#############################################################################
//  File:      PerPixBlinn.vert
//  Purpose:   GLSL vertex program for per fragment Blinn-Phong lighting
//  Author:    Marcus Hudritsch
//  Date:      February 2014
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

attribute   vec4  a_position;    // Vertex position attribute
attribute   vec3  a_normal;      // Vertex normal attribute
attribute   vec2  a_texCoord;
attribute   vec4  a_color;
attribute   vec3  a_tangent;
in		    ivec4 a_boneIds;
attribute   vec4  a_boneWeights;

const       int   MAX_BONES = 100;
uniform		mat4  u_bones[MAX_BONES];

uniform     mat4  u_mvMatrix;    // modelview matrix 
uniform     mat3  u_nMatrix;     // normal matrix=transpose(inverse(mv))
uniform     mat4  u_mvpMatrix;   // = projection * modelView

varying     vec3  v_P_VS;        // Point of illumination in view space (VS)
varying     vec3  v_N_VS;        // Normal at P_VS in view space
varying     vec2  v_texCoord;    // Texture coordiante varying

//-----------------------------------------------------------------------------
void main(void)
{  
	mat4 boneTransform = 	u_bones[a_boneIds.x] * a_boneWeights.x
						+ 	u_bones[a_boneIds.y] * a_boneWeights.y
						+ 	u_bones[a_boneIds.z] * a_boneWeights.z
						+ 	u_bones[a_boneIds.w] * a_boneWeights.w;

   v_P_VS = vec3(u_mvMatrix * boneTransform * a_position);
   v_N_VS = vec3(u_nMatrix * transpose(inverse(mat3(boneTransform))) * a_normal);  
   v_N_VS = normalize(v_N_VS);
   v_texCoord = a_texCoord;
   gl_Position = u_mvpMatrix * boneTransform * a_position;
}
//-----------------------------------------------------------------------------
