//#############################################################################
//  File:      BumpNormal.vert
//  Purpose:   GLSL normal map bump mapping
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision mediump float;
#endif

//-----------------------------------------------------------------------------
attribute   vec4  a_position; // Vertex position attribute
attribute   vec3  a_normal;   // Vertex normal attribute
attribute   vec4  a_tangent;  // Vertex tangent attribute
attribute   vec2  a_texCoord; // Vertex texture coordiante attribute

uniform     mat4  u_mvMatrix;   // modelview matrix 
uniform     mat3  u_nMatrix;    // normal matrix=transpose(inverse(mv))
uniform     mat4  u_mvpMatrix;  // = projection * modelView

uniform     vec4  u_lightPosVS[8];     // position of light in view space
uniform     vec3  u_lightDirVS[8];     // spot direction in view space
uniform     float u_lightSpotCutoff[8];// spot cutoff angle 1-180 degrees

varying     vec2  v_texCoord; // Texture coordiante varying
varying     vec3  v_L_TS;     // Vector to the light 0 in tangent space
varying     vec3  v_E_TS;     // Vector to the eye in tangent space
varying     vec3  v_S_TS;     // Spot direction in tangent space
varying     float v_d;        // Light distance

void main()
{  
    // Pass the texture coord. for interpolation
    v_texCoord = a_texCoord;
   
    // Building the matrix Eye Space -> Tangent Space
    // See the math behind at: http://www.terathon.com/code/tangent.html
    vec3 n = normalize(u_nMatrix * a_normal);
    vec3 t = normalize(u_nMatrix * a_tangent.xyz);
    vec3 b = cross(n, t) * a_tangent.w; // bitangent w. corrected handedness
    mat3 TBN = mat3(t,b,n);
   
    // Transform vertex into view space
    vec3 P_VS = vec3(u_mvMatrix *  a_position);
   
    // Transform spotdir into tangent space
    if (u_lightSpotCutoff[0] < 180.0)
    {   v_S_TS = u_lightDirVS[0];
        v_S_TS *= TBN;
    }
      
    // Transform vector to the light 0 into tangent space
    vec3 L = u_lightPosVS[0].xyz - P_VS;
    v_d = length(L);  // calculate distance to light before normalizing
    v_L_TS = L;
    v_L_TS *= TBN;
   
    // Transform vector to the eye into tangent space
    v_E_TS = -P_VS;  // eye vector in view space
    v_E_TS *= TBN;
     
    // pass the vertex w. the fix-function transform
    gl_Position = u_mvpMatrix * a_position;
}
