//#############################################################################
//  File:      RefractReflect.frag
//  Purpose:   GLSL fragment program for refraction- & reflection mapping
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec3        v_R_OS;                 // Reflected ray in object space
in      vec3        v_T_OS;                 // Refracted ray in object space
in      float       v_F_Theta;              // Fresnel reflection coefficient
in      vec4        v_specColor;            // Specular color at vertex

uniform vec4        u_matDiff;              // diffuse color refl. coefficient (kd)
uniform samplerCube u_matTextureDiffuse0;   // Cubic environment texture map
uniform float       u_oneOverGamma;         // 1.0f / Gamma correction value

out     vec4        o_fragColor;            // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
    // get the reflection & refraction color out of the cubic map
    vec4 reflCol = texture(u_matTextureDiffuse0, v_R_OS);
    vec4 refrCol = texture(u_matTextureDiffuse0, v_T_OS);
   
    // Mix the final color with the fast frenel factor
    o_fragColor = mix(refrCol, reflCol, v_F_Theta);
    
    // Add diffuse color as transmission
    o_fragColor.rgb += u_matDiff.rgb;
    
    // Add specular highlight
    o_fragColor.rgb += v_specColor.rgb;
   
    // For correct alpha blending overwrite alpha component
    o_fragColor.a = 1.0 - u_matDiff.a;

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
