//#############################################################################
//  File:      Terrain.frag
//  Purpose:   GLSL per vertex diffuse lighting with texturing
//  Author:    Marcus Hudritsch
//  Date:      September 2012 (HS12)
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec4      v_color;        // Interpol. ambient & diff. color
in      vec2      v_texCoord;     // Interpol. texture coordinate

uniform sampler2D u_matTexture0;     // Color map

out     vec4      o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
void main()
{  // Interpolated ambient & diffuse components  
   o_fragColor = v_color;
   
   // componentwise multiply w. texture color
   o_fragColor *= texture(u_matTexture0, v_texCoord);
}
//-----------------------------------------------------------------------------