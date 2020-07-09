//#############################################################################
//  File:      Reflect.frag
//  Purpose:   GLSL fragment program for reflection mapping
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
in      vec3        v_R_OS;         // Reflected ray in object space
in      vec4        v_specColor;    // Specular color at vertex

uniform vec4        u_matAmbient;   // ambient color reflection coefficient (ka)
uniform vec4        u_matDiffuse;   // diffuse color reflection coefficient (kd)
uniform vec4        u_matSpecular;  // specular color reflection coefficient (ks)
uniform vec4        u_matEmissive;  // emissive color for self-shining materials
uniform float       u_matShininess; // shininess exponent
uniform float       u_oneOverGamma; // 1.0f / Gamma correction value
uniform samplerCube u_texture0;     // Cubic environment texture map

out     vec4        o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
    // Get the reflection & refraction color out of the cubic map
    o_fragColor = texture(u_texture0, v_R_OS);
   
    // Add Specular highlight
    o_fragColor.rgb += v_specColor.rgb;
   
    // For correct alpha blending overwrite alpha component
    o_fragColor.a = 1.0-u_matDiffuse.a;

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
