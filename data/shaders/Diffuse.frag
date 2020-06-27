//#############################################################################
//  File:      Diffuse.frag
//  Purpose:   GLSL fragment program for simple diffuse per vertex lighting
//  Date:      September 2012 (HS12)
//  Copyright: Marcus Hudritsch
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

//-----------------------------------------------------------------------------
in       vec4   diffuseColor;   // interpolated color calculated in the vertex shader

uniform  float  u_oneOverGamma; // 1.0f / Gamma correction value

out      vec4   o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
void main()
{     
   o_fragColor = diffuseColor;

   // Apply gamma correction
   o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
