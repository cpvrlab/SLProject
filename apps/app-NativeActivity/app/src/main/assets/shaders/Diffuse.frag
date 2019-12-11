//#############################################################################
//  File:      Diffuse.frag
//  Purpose:   GLSL fragment program for simple diffuse per vertex lighting
//  Date:      September 2012 (HS12)
//  Copyright: Marcus Hudritsch
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

varying vec4 diffuseColor;      // interpolated color calculated in the vertex shader
uniform float u_oneOverGamma;   // 1.0f / Gamma correction value

void main()
{     
   gl_FragColor = diffuseColor;

   // Apply gamma correction
   gl_FragColor.rgb = pow(gl_FragColor.rgb, vec3(u_oneOverGamma));
}
