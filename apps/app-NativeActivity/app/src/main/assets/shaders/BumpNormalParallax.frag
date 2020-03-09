//#############################################################################
//  File:      BumpNormalParallax.frag
//  Purpose:   GLSL parallax normal bump mapping
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision mediump float;
#endif

uniform bool   u_lightIsOn[8];      // flag if light is on
uniform vec4   u_lightPosVS[8];     // position of light in view space
uniform vec4   u_lightAmbient[8];   // ambient light intensity (Ia)
uniform vec4   u_lightDiffuse[8];   // diffuse light intensity (Id)
uniform vec4   u_lightSpecular[8];  // specular light intensity (Is)
uniform vec3   u_lightSpotDirVS[8]; // spot direction in view space
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
uniform float  u_oneOverGamma;      // 1.0f / Gamma correction value

uniform sampler2D u_texture0;       // Color map
uniform sampler2D u_texture1;       // Normal map
uniform sampler2D u_texture2;       // Height map;
uniform float     u_scale;          // Height scale for parallax mapping
uniform float     u_offset;         // Height bias for parallax mapping

varying vec2   v_texCoord;          // Texture coordiante varying
varying vec3   v_L_TS;              // Vector to the light in tangent space
varying vec3   v_E_TS;              // Vector to the eye in tangent space
varying vec3   v_S_TS;              // Spot direction in tangent space
varying float  v_d;                 // Light distance

void main()
{
    vec3 E = normalize(v_E_TS);   // normalized eye direction
    vec3 L = normalize(v_L_TS);   // normalized light direction
    vec3 S;                       // normalized spot direction
   
    // Halfvector H between L & E (See Blinn's lighting model)
    vec3 H = normalize(L + E);
   
    ////////////////////////////////////////////////////////////
    // Calculate new texture coord. Tc for Parallax mapping
    // The height comes from red channel from the height map
    float height = texture2D(u_texture2, v_texCoord.st).r;
   
    // Scale the height and add the bias (height offset)
    height = height * u_scale + u_offset;
   
    // Add the texture offset to the texture coord.
    vec2 Tc = v_texCoord.st + (height * E.st);
    ////////////////////////////////////////////////////////////
   
    // Get normal from normal map, move from [0,1] to [-1, 1] range & normalize
    //vec3 N = vec3(0.0,0.0,1.0);
    vec3 N = normalize(texture2D(u_texture1, Tc).rgb * 2.0 - 1.0);
   
    // Calculate attenuation over distance v_d
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
    {   S = normalize(v_S_TS);
        float spotDot; // Cosine of angle between L and spotdir
        float spotAtt; // Spot attenuation
        spotDot = dot(-L, S);
        if (spotDot < u_lightSpotCosCut[0]) spotAtt = 0.0;
        else spotAtt = max(pow(spotDot, u_lightSpotExp[0]), 0.0);
        att *= spotAtt;
    }
   
    // compute diffuse lighting
    float diffFactor = max(dot(L,N), 0.0) ;
   
    // compute specular lighting
    float specFactor = pow(max(dot(N,H), 0.0), u_matShininess);
   
    // add ambient & diffuse light components
    gl_FragColor = u_matEmissive + 
                   u_globalAmbient +
                   att * u_lightAmbient[0] * u_matAmbient +
                   att * u_lightDiffuse[0] * u_matDiffuse * diffFactor;
   
    // componentwise multiply w. texture color
    gl_FragColor *= texture2D(u_texture0, Tc);
   
    // add finally the specular part
    gl_FragColor += att * u_lightSpecular[0] * u_matSpecular * specFactor;

    // Apply gamma correction
    gl_FragColor.rgb = pow(gl_FragColor.rgb, vec3(u_oneOverGamma));
}
