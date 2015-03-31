//#############################################################################
//  File:      PerVrtBlinnTexSkinned.vert
//  Purpose:   GLSL vertex program for per vertex Blinn-Phong lighting with
//             vertex skinning for skeleton animation
//  Author:    Marc Wacker, Marcus Hudritsch
//  Date:      July 2015
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#version 150

//-----------------------------------------------------------------------------
attribute vec4 a_position;          // Vertex position attribute
attribute vec3 a_normal;            // Vertex normal attribute
attribute vec2 a_texCoord;          // Vertex texture coord. attribute
attribute vec4 a_jointIds;          // Vertex joint indexes attributes
attribute vec4 a_jointWeights;      // Vertex joint weights attributes

uniform mat4   u_mvMatrix;          // modelview matrix 
uniform mat3   u_nMatrix;           // normal matrix=transpose(inverse(mv))
uniform mat4   u_mvpMatrix;         // = projection * modelView
uniform mat4   u_jointMatrices[100];// joint matrices for vertex skinning

uniform int    u_numLightsUsed;     // NO. of lights used light arrays
uniform bool   u_lightIsOn[8];      // flag if light is on
uniform vec4   u_lightPosVS[8];     // position of light in view space
uniform vec4   u_lightAmbient[8];   // ambient light intensity (Ia)
uniform vec4   u_lightDiffuse[8];   // diffuse light intensity (Id)
uniform vec4   u_lightSpecular[8];  // specular light intensity (Is)
uniform vec3   u_lightDirVS[8];     // spot direction in view space
uniform float  u_lightSpotCutoff[8];// spot cutoff angle 1-180 degrees
uniform float  u_lightSpotCosCut[8];// cosine of spot cutoff angle
uniform float  u_lightSpotExp[8];   // spot exponent
uniform vec3   u_lightAtt[8];       // attenuation (const,linear,quadr.)
uniform bool   u_lightDoAtt[8];     // flag if att. must be calc.
uniform vec4   u_globalAmbient;     // Global ambient scene color

uniform vec4   u_matAmbient;        // ambient color reflection coefficient (ka)
uniform vec4   u_matDiffuse;        // diffuse color reflection coefficient (kd)
uniform vec4   u_matSpecular;       // specular color reflection coefficient (ks)
uniform vec4   u_matEmissive;       // emissive color for selfshining materials
uniform float  u_matShininess;      // shininess exponent

varying vec4   v_color;             // Ambient & diffuse color at vertex
varying vec4   v_specColor;         // Specular color at vertex
varying vec2   v_texCoord;          // texture coordinate at vertex

//-----------------------------------------------------------------------------
void PointLight (in    int  i,   // OpenGL light number
                 in    vec3 P_VS,// Point of illumination in VS
                 in    vec3 N,   // Normalized normal at P_VS
                 in    vec3 E,   // Normalized vector from P_VS to view in VS
                 inout vec4 Ia,  // Ambient light intesity
                 inout vec4 Id,  // Diffuse light intesity
                 inout vec4 Is)  // Specular light intesity
{  
    // Vector from P_VS to the light in VS
    vec3 L = u_lightPosVS[i].xyz - P_VS;
      
    // Calculate attenuation over distance & normalize L
    float att = 1.0;
    if (u_lightDoAtt[i])
    {   vec3 att_dist;
        att_dist.x = 1.0;
        att_dist.z = dot(L,L);         // = distance * distance
        att_dist.y = sqrt(att_dist.z); // = distance
        att = 1.0 / dot(att_dist, u_lightAtt[i]);
        L /= att_dist.y;               // = normalize(L)
    } else L = normalize(L);
   
    // Normalized halfvector between N and L
    vec3 H = normalize(L+E);
   
    // Calculate diffuse & specular factors
    float diffFactor = max(dot(N,L), 0.0);
    float specFactor = 0.0;
    if (diffFactor!=0.0) 
        specFactor = pow(max(dot(N,H), 0.0), u_matShininess);
   
    // Calculate spot attenuation
    if (u_lightSpotCutoff[i] < 180.0)
    {   float spotDot; // Cosine of angle between L and spotdir
        float spotAtt; // Spot attenuation
        spotDot = dot(-L, u_lightDirVS[i]);
        if (spotDot < u_lightSpotCosCut[i]) spotAtt = 0.0;
        else spotAtt = max(pow(spotDot, u_lightSpotExp[i]), 0.0);
        att *= spotAtt;
    }
   
    // Accumulate light intesities
    Ia += att * u_lightAmbient[i];
    Id += att * u_lightDiffuse[i] * diffFactor;
    Is += att * u_lightSpecular[i] * specFactor;
}
//-----------------------------------------------------------------------------
void main()
{
    vec4 Ia, Id, Is;        // Accumulated light intensities at P_VS
    Ia = vec4(0.0);         // Ambient light intesity
    Id = vec4(0.0);         // Diffuse light intesity
    Is = vec4(0.0);         // Specular light intesity
   
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
                        
   vec3 P_VS = vec3(u_mvMatrix * jm * a_position);
   vec3 N = normalize(vec3(u_nMatrix * jnm * a_normal));
   vec3 E = normalize(-P_VS);

    // Early versions of GLSL do not allow uniforms in for loops
    for (int i=0; i<8; i++)
    {   // check if light is used and on
        if (i < u_numLightsUsed && u_lightIsOn[i])
            PointLight(i, P_VS, N, E, Ia, Id, Is);
    }
   
    // Set the texture coord. varying for interpolated tex. coords.
    v_texCoord = a_texCoord.xy;
   
    // Sum up all the reflected color components except the specular
    v_color =  u_matEmissive +
               u_globalAmbient +
               Ia * u_matAmbient +
               Id * u_matDiffuse;
   
    // Calculate the specular reflection separately 
    v_specColor =  Is * u_matSpecular;

    // For correct alpha blending overwrite alpha component
    v_color.a = u_matDiffuse.a;

    // Transform the vertex with the modelview and joint matrix
    gl_Position = u_mvpMatrix * jm * a_position;
}
//-----------------------------------------------------------------------------
