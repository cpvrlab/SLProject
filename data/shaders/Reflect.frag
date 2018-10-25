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
uniform vec4   u_matAmbient;        // ambient color reflection coefficient (ka)
uniform vec4   u_matDiffuse;        // diffuse color reflection coefficient (kd)
uniform vec4   u_matSpecular;       // specular color reflection coefficient (ks)
uniform vec4   u_matEmissive;       // emissive color for selfshining materials
uniform float  u_matShininess;      // shininess exponent

uniform samplerCube  u_texture0;    // Cubic environment texture map

varying vec3   v_R_OS;              // Reflected ray in object space
varying vec4   v_specColor;         // Specular color at vertex

//-----------------------------------------------------------------------------
void main()
{     
    // Get the reflection & refraction color out of the cubic map
    gl_FragColor = textureCube(u_texture0, v_R_OS);
   
    // Add Specular highlight
    gl_FragColor.rgb += v_specColor.rgb;
   
    // For correct alpha blending overwrite alpha component
    gl_FragColor.a = 1.0-u_matDiffuse.a;
}
//-----------------------------------------------------------------------------