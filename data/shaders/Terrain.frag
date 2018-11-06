//#############################################################################
//  File:      Terrain.frag
//  Purpose:   GLSL per vertex diffuse lighting with texturing
//  Author:    Marcus Hudritsch
//  Date:      September 2012 (HS12)
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision mediump float;
#endif

//-----------------------------------------------------------------------------
varying vec4      v_color;             // Interpol. ambient & diff. color
varying vec2      v_texCoord;          // Interpol. texture coordinate

uniform sampler2D u_texture0;          // Color map

//-----------------------------------------------------------------------------
void main()
{  // Interpolated ambient & diffuse components  
   gl_FragColor = v_color;
   
   // componentwise multiply w. texture color
   gl_FragColor *= texture2D(u_texture0, v_texCoord);
}
//-----------------------------------------------------------------------------