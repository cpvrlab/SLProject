//#############################################################################
//  File:      PerPixBlinnTex.frag
//  Purpose:   GLSL per pixel lighting with texturing
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
varying vec3   v_P_VS;              // Interpol. point of illum. in view space (VS)
varying vec3   v_N_VS;              // Interpol. normal at v_P_VS in view space
varying vec2   v_texCoord;          // Interpol. texture coordinate in tex. space

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

uniform int    u_projection;        // type of stereo
uniform int    u_stereoEye;         // -1=left, 0=center, 1=right 
uniform mat3   u_stereoColorFilter; // color filter matrix
uniform sampler2D u_texture0;       // Color map

//-----------------------------------------------------------------------------
void PointLight (in    int  i,   // OpenGL light number
                 in    vec3 P_VS,// Point of illumination in VS
                 in    vec3 N,   // Normalized normal at v_P_VS
                 in    vec3 E,   // Normalized vector from v_P_VS to view in VS
                 inout vec4 Ia,  // Ambient light intesity
                 inout vec4 Id,  // Diffuse light intesity
                 inout vec4 Is)  // Specular light intesity
{  
    // Vector from v_P_VS to the light in VS
    vec3 L = u_lightPosVS[i].xyz - v_P_VS;
      
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
    vec4 Ia, Id, Is;  // Accumulated light intensities at v_P_VS
   
    Ia = vec4(0.0);         // Ambient light intesity
    Id = vec4(0.0);         // Diffuse light intesity
    Is = vec4(0.0);         // Specular light intesity
   
    vec3 N = normalize(v_N_VS);  // A varying normal has not anymore unit length
    vec3 E = normalize(-v_P_VS);  // Vector from p to the eye
   
    // Early versions of GLSL do not allow uniforms in for loops
    for (int i=0; i<8; i++)
    {   if (i < u_numLightsUsed && u_lightIsOn[i])
        {  PointLight (i, v_P_VS, N, E, Ia, Id, Is);
        }
    }
   
    // Sum up all the reflected color components
    gl_FragColor =  u_globalAmbient +
                    u_matEmissive + 
                    Ia * u_matAmbient +
                    Id * u_matDiffuse;
                   
    // Componentwise multiply w. texture color
    gl_FragColor *= texture2D(u_texture0, v_texCoord); 
   
    // add finally the specular RGB-part
    vec4 specColor = Is * u_matSpecular;
    gl_FragColor.rgb += specColor.rgb;
   
    // Apply stereo eye separation
    if (u_projection > 1)
    {   if (u_projection > 7) // stereoColor??
        {   // Apply color filter but keep alpha
            gl_FragColor.rgb = u_stereoColorFilter * gl_FragColor.rgb;
        }
        else if (u_projection == 5) // stereoLineByLine
        {   if (mod(floor(gl_FragCoord.y), 2.0) < 0.5) // even
            {  if (u_stereoEye ==-1) discard;
            } else // odd
            {  if (u_stereoEye == 1) discard;
            }
        }
        else if (u_projection == 6) // stereoColByCol
        {   if (mod(floor(gl_FragCoord.x), 2.0) < 0.5) // even
            {  if (u_stereoEye ==-1) discard;
            } else // odd
            {  if (u_stereoEye == 1) discard;
            }
        } 
        else if (u_projection == 7) // stereoCheckerBoard
        {   bool h = (mod(floor(gl_FragCoord.x), 2.0) < 0.5);
            bool v = (mod(floor(gl_FragCoord.y), 2.0) < 0.5);
            if (h==v) // both even or odd
            {   if (u_stereoEye ==-1) discard;
            } else // odd
            {   if (u_stereoEye == 1) discard;
            }
        }
    }
}
//-----------------------------------------------------------------------------
