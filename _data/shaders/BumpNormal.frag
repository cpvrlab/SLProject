//#############################################################################
//  File:      BumpNormal.frag
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

uniform bool   u_lightIsOn[8];      //! flag if light is on
uniform vec4   u_lightPosVS[8];     //! position of light in view space
uniform vec4   u_lightAmbient[8];   //! ambient light intensity (Ia)
uniform vec4   u_lightDiffuse[8];   //! diffuse light intensity (Id)
uniform vec4   u_lightSpecular[8];  //! specular light intensity (Is)
uniform vec3   u_lightDirVS[8];     //! spot direction in view space
uniform float  u_lightSpotCosCut[8];//! cosine of spot cutoff angle
uniform float  u_lightSpotCutoff[8];//! spot cutoff angle 1-180 degrees
uniform float  u_lightSpotExp[8];   //! spot exponent
uniform vec3   u_lightAtt[8];       //! attenuation (const,linear,quadr.)
uniform bool   u_lightDoAtt[8];     //! flag if att. must be calc.
uniform vec4   u_globalAmbient;     //! Global ambient scene color

uniform vec4   u_matAmbient;        //! ambient color reflection coefficient (ka)
uniform vec4   u_matDiffuse;        //! diffuse color reflection coefficient (kd)
uniform vec4   u_matSpecular;       //! specular color reflection coefficient (ks)
uniform vec4   u_matEmissive;       //! emissive color for selfshining materials
uniform float  u_matShininess;      //! shininess exponent

uniform sampler2D u_texture0;       //! Color map
uniform sampler2D u_texture1;       //! Normal map

varying vec2   v_texCoord;          //! Texture coordiante varying
varying vec3   v_L_TS;              //! Vector to light 0 in tangent space
varying vec3   v_E_TS;              //! Vector to the eye in tangent space
varying vec3   v_S_TS;              //! Spot direction in tangent space
varying float  v_d;                 //! Light distance

//! Fragment shader for classic normal map bump mapping
void main()
{   
    vec3 E = normalize(v_E_TS);   // normalized eye direction
    vec3 L = normalize(v_L_TS);   // normalized light direction
   
    // Halfvector H between L & E (See Blinn's lighting model)
    vec3 H = normalize(L + E);
  
    // Get normal from normal map, move from [0,1] to [-1, 1] range & normalize
    vec3 N = normalize(texture2D(u_texture1, v_texCoord).rgb * 2.0 - 1.0);
   
    // Calculate attenuation over distance d
    float att = 1.0;  // Total attenuation factor
    if (u_lightDoAtt[0])
    {   vec3 att_dist;
        att_dist.x = 1.0;
        att_dist.y = v_d;
        att_dist.z = v_d * v_d;
        att = 1.0 / dot(att_dist, u_lightAtt[0]);
    }
   
    // Calculate spot attenuation
    if (u_lightSpotCutoff[0] < 180.0)
    {   vec3 S = normalize(v_S_TS); // normalized spot direction
        float spotDot; // Cosine of angle between L and spotdir
        float spotAtt; // Spot attenuation
        spotDot = dot(-L, S);
        if (spotDot < u_lightSpotCosCut[0]) spotAtt = 0.0;
        else spotAtt = max(pow(spotDot, u_lightSpotExp[0]), 0.0);
        att *= spotAtt;
    }
   
    // Compute diffuse lighting
    float diffFactor = max(dot(L,N), 0.0) ;
   
    // Compute specular lighting
    float specFactor = pow(max(dot(N,H), 0.0), u_matShininess);
   
    // Add ambient & diffuse light components
    gl_FragColor = u_matEmissive + 
                   u_globalAmbient +
                   att * u_lightAmbient[0] * u_matAmbient +
                   att * u_lightDiffuse[0] * u_matDiffuse * diffFactor;
   
    // Componentwise multiply w. texture color
    gl_FragColor *= texture2D(u_texture0, v_texCoord);
   
    // Add finally the specular part
    gl_FragColor += att * u_lightSpecular[0] * u_matSpecular * specFactor;
}
