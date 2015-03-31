//#############################################################################
//  File:      ADS.frag
//  Purpose:   GLSL fragment program for simple ADS per vertex lighting with
//             texture mapping
//  Author:    Marcus Hudritsch
//  Date:      September 2012 (HS12)
//  Copyright: Marcus Hudritsch
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision mediump float;
#endif

varying vec4      v_color;      // interpolated color calculated in the vertex shader 
varying vec4      v_specColor;  // interpolated specular color 
varying vec2      v_texCoord;   // interpolated texture coordinate

uniform sampler2D u_texture0;   // texture map
uniform sampler2D u_texture1;   // gloss map

void main()
{  
   // Just set the interpolated color from the vertex shader
   gl_FragColor = v_color;

   // componentwise multiply w. texture color
   gl_FragColor *= texture2D(u_texture0, v_texCoord);

   // add componentwise the specular part multiplied by the gloss maps r channel
   gl_FragColor += v_specColor * texture2D(u_texture1, v_texCoord).r;
}
