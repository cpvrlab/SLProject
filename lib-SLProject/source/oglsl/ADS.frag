//#############################################################################
//  File:      ADS.frag
//  Purpose:   GLSL fragment program for simple ADS per vertex lighting
//  Author:    Marcus Hudritsch
//  Date:      September 2011 (HS11)
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision mediump float;
#endif

varying vec4 v_color;   // interpolated color calculated in the vertex shader 

void main()
{     
   gl_FragColor = v_color;
}