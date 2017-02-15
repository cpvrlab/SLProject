//#############################################################################
//  File:      Diffuse.frag
//  Purpose:   GLSL fragment program for simple diffuse per vertex lighting
//  Date:      September 2012 (HS12)
//  Copyright: Marcus Hudritsch
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

varying vec4 diffuseColor;   // interpolated color calculated in the vertex shader

void main()
{     
   gl_FragColor = diffuseColor;
}
