//#############################################################################
//  File:      ADSTex.frag
//  Purpose:   GLSL fragment program for simple ADS per vertex lighting with
//             texture mapping
//  Author:    Marcus Hudritsch
//  Date:      September 2011 (HS11)
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#ifdef GL_ES
precision mediump float;
#endif

varying vec4      v_color;      // interpolated color from the vertex shader
varying vec2      v_texCoord;   // interpolated texture coordinate

uniform sampler2D u_texture0;   // texture map

void main()
{  
   // Just set the interpolated color from the vertex shader
   gl_FragColor = v_color;

   // componentwise multiply w. texture color
   gl_FragColor *= texture2D(u_texture0, v_texCoord);
}
