//#############################################################################
//  File:      PerPixBlinnTexNrm.vert
//  Purpose:   GLSL normal map bump mapping
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces #Lights by SLVLights.size()
#pragma define NUM_LIGHTS #Lights
//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;  // Vertex position attribute
layout (location = 1) in vec3  a_normal;    // Vertex normal attribute
layout (location = 2) in vec2  a_uv1;       // Vertex texture coordiante attribute
layout (location = 5) in vec4  a_tangent;   // Vertex tangent attribute

uniform mat4  u_mvMatrix;   // modelview matrix
uniform mat3  u_nMatrix;    // normal matrix=transpose(inverse(mv))
uniform mat4  u_mvpMatrix;  // = projection * modelView

uniform vec4  u_lightPosVS[NUM_LIGHTS];     // position of light in view space
uniform vec3  u_lightSpotDir[NUM_LIGHTS];   // spot direction in view space
uniform float u_lightSpotDeg[NUM_LIGHTS];   // spot cutoff angle 1-180 degrees

out     vec3  v_P_VS;                   // Point of illumination in view space (VS)
out     vec2  v_uv1;                    // Texture coordiante output
out     vec3  v_eyeDirTS;               // Vector to the eye in tangent space
out     vec3  v_lightDirTS[NUM_LIGHTS]; // Vector to the light 0 in tangent space
out     vec3  v_spotDirTS[NUM_LIGHTS];  // Spot direction in tangent space
out     float v_lightDist[NUM_LIGHTS];  // Light distance
//-----------------------------------------------------------------------------
void main()
{  
    // Pass the texture coord. for interpolation
    v_uv1 = a_uv1;
   
    // Building the matrix Eye Space -> Tangent Space
    // See the math behind at: http://www.terathon.com/code/tangent.html
    vec3 n = normalize(u_nMatrix * a_normal);
    vec3 t = normalize(u_nMatrix * a_tangent.xyz);
    vec3 b = cross(n, t) * a_tangent.w; // bitangent w. corrected handedness
    mat3 TBN = mat3(t,b,n);
   
    // Transform vertex into view space
    v_P_VS = vec3(u_mvMatrix *  a_position);

    // Transform vector to the eye into tangent space
    v_eyeDirTS = -v_P_VS;  // eye vector in view space
    v_eyeDirTS *= TBN;

    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        // Transform spotdir into tangent space
        v_spotDirTS[i] = u_lightSpotDir[i];
        v_spotDirTS[i]  *= TBN;

        // Transform vector to the light 0 into tangent space
        vec3 L = u_lightPosVS[i].xyz - v_P_VS;
        v_lightDist[i]  = length(L);  // calculate distance to light before normalizing
        v_lightDirTS[i] = L;
        v_lightDirTS[i] *= TBN;
    }

    // pass the vertex w. the fix-function transform
    gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
