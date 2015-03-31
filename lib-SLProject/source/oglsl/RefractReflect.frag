//#############################################################################
//  File:      RefractReflect.frag
//  Purpose:   GLSL fragment program for refraction- & reflection mapping
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
uniform vec4   u_matDiffuse;        // diffuse color reflection coefficient (kd)

uniform samplerCube  u_texture0;    // Cubic environment texture map

varying vec3   v_R_OS;              // Reflected ray in object space
varying vec3   v_T_OS;              // Refracted ray in object space
varying float  v_F_Theta;           // Fresnel reflection coefficient
varying vec4   v_specColor;         // Specular color at vertex

//-----------------------------------------------------------------------------
void main()
{     
    // get the reflection & refraction color out of the cubic map
    vec4 reflCol = textureCube(u_texture0, v_R_OS);
    vec4 refrCol = textureCube(u_texture0, v_T_OS);
   
    // Mix the final color with the fast frenel factor
    gl_FragColor = mix(refrCol, reflCol, v_F_Theta);

    // Add specular highlight
    gl_FragColor.rgb += v_specColor.rgb;
   
    // For correct alpha blending overwrite alpha component
    gl_FragColor.a = 1.0 - u_matDiffuse.a;
}
//-----------------------------------------------------------------------------