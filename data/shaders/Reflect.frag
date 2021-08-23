//#############################################################################
//  File:      Reflect.frag
//  Purpose:   GLSL fragment program for reflection mapping
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

//precision highp float;

//-----------------------------------------------------------------------------
in      vec3        v_R_OS;         // Reflected ray in object space
in      vec4        v_specColor;    // Specular color at vertex

uniform samplerCube u_matTextureDiffuse0;  // Cubic environment texture map
uniform vec4        u_matAmbi;             // ambient color reflection coefficient (ka)
uniform vec4        u_matDiff;             // diffuse color reflection coefficient (kd)
uniform vec4        u_matSpec;             // specular color reflection coefficient (ks)
uniform vec4        u_matEmis;             // emissive color for self-shining materials
uniform float       u_matShin;             // shininess exponent
uniform float       u_oneOverGamma;        // 1.0f / Gamma correction value
out     vec4        o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
    // Get the reflection & refraction color out of the cubic map
    vec4 col = texture(u_matTextureDiffuse0, v_R_OS);
    o_fragColor = col;
    
    // Add Specular highlight
    o_fragColor.rgb += v_specColor.rgb;
   
    // For correct alpha blending overwrite alpha component
    o_fragColor.a = 1.0-u_matDiff.a;

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
