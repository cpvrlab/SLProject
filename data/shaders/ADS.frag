//#############################################################################
//  File:      ADS.frag
//  Purpose:   GLSL fragment program for simple ADS per vertex lighting
//  Author:    Marcus Hudritsch
//  Date:      September 2011 (HS11)
//  Copyright: Marcus Hudritsch
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in       vec4   v_color;      // interpolated color calculated in the vertex shader

out      vec4   o_fragColor;  // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
   o_fragColor = v_color;
}
//-----------------------------------------------------------------------------