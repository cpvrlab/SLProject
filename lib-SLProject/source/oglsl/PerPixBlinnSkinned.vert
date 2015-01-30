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
attribute   vec2  a_texCoord;    // Vertex texture attribute
attribute   vec4  a_color;       // Vertex color attribute
attribute   vec3  a_tangent;     // Vertex tangent attribute
attribute   vec4  a_jointIds;    // Vertex joint indexes attributes
attribute   vec4  a_jointWeights;// Vertex joint weights attributes

uniform     mat4  u_mvMatrix;    // modelview matrix 
uniform     mat3  u_nMatrix;     // normal matrix=transpose(inverse(mv))
uniform     mat4  u_mvpMatrix;   // = projection * modelView
uniform     mat4  u_jointMatrices[100]; // joint matrices for skinning

varying     vec3  v_P_VS;        // Point of illumination in view space (VS)
varying     vec3  v_N_VS;        // Normal at P_VS in view space
varying     vec2  v_texCoord;    // Texture coordiante varying

//-----------------------------------------------------------------------------
void main(void)
{  
    // In skinned skeleton animation every vertex of a mesh is transformed by
    // max. four joints (bones) of a skeleton identified by indexes. The joint
    // matrix is a weighted sum fore joint matrices and can change per frame
    // to animate the mesh
    mat4 jm = u_jointMatrices[int(a_jointIds.x)] * a_jointWeights.x
            + u_jointMatrices[int(a_jointIds.y)] * a_jointWeights.y
            + u_jointMatrices[int(a_jointIds.z)] * a_jointWeights.z
            + u_jointMatrices[int(a_jointIds.w)] * a_jointWeights.w;

    // Build the 3x3 submatrix in GLSL 110 (= mat3 jt3 = mat3(jt))
    // for the normal transform that is the normally the inverse transpose.
    // The inverse transpose can be ignored as long as we only have
    // rotation and uniform scaling in the 3x3 submatrix.
    mat3 jnm;
    jnm[0][0] = jm[0][0]; jnm[1][0] = jm[1][0]; jnm[2][0] = jm[2][0];
    jnm[0][1] = jm[0][1]; jnm[1][1] = jm[1][1]; jnm[2][1] = jm[2][1];
    jnm[0][2] = jm[0][2]; jnm[1][2] = jm[1][2]; jnm[2][2] = jm[2][2];

   v_P_VS = vec3(u_mvMatrix * jm * a_position);
   v_N_VS = vec3(u_nMatrix * jnm * a_normal);
   v_N_VS = normalize(v_N_VS);
   v_texCoord = a_texCoord;

   // Transform the vertex with the modelview and joint matrix
   gl_Position = u_mvpMatrix * jm * a_position;
}
//-----------------------------------------------------------------------------
