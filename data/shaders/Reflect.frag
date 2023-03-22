//#############################################################################
//  File:      Reflect.frag
//  Purpose:   GLSL fragment program for reflection mapping
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec3        v_R_OS;                 // Reflected ray in object space
in      vec4        v_specColor;            // Specular color at vertex

uniform vec4        u_matDiff;              // diffuse color refl. coefficient (kd)
uniform samplerCube u_matTextureDiffuse0;   // Cubic environment texture map
uniform float       u_oneOverGamma;         // 1.0f / Gamma correction value

out     vec4        o_fragColor;            // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
    // Get the reflection & refraction color out of the cubic map
    o_fragColor = texture(u_matTextureDiffuse0, v_R_OS);

    // Add Specular highlight
    o_fragColor.rgb += v_specColor.rgb;

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
