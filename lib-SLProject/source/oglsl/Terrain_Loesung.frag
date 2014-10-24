//#############################################################################
//  File:      Terrain.frag
//  Purpose:   GLSL per vertex diffuse lighting with texturing
//  Author:    Marcus Hudritsch
//  Date:      September 2011 (HS11
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

//Required to be able to use bitwise operators
//#version 130

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#endif


//-----------------------------------------------------------------------------
varying vec4      v_color;             // Interpol. ambient & diff. color
varying vec2      v_texCoord;          // Interpol. texture coordinate

uniform sampler2D u_texture0;          // Color map
uniform sampler2D u_texture2;          // Vector map
uniform int       u_overlayOn1;
uniform int       u_overlayOn2;
uniform int       u_overlayOn3;
uniform int       u_overlayOn4;
uniform int       u_overlayOn5;
uniform int       u_overlayOn6;
uniform int       u_overlayOn7;
uniform int       u_overlayOn8;

//-----------------------------------------------------------------------------
void main()
{  // Interpolated ambient & diffuse components
   gl_FragColor = v_color;
   
   // componentwise multiply w. texture color
   gl_FragColor *= texture2D(u_texture0, v_texCoord);

   ivec4 iInfo = ivec4(255.0 * texture2D(u_texture2, v_texCoord));
   
	gl_FragColor += float(iInfo.r &   1) * u_overlayOn1 * vec4(1.0, 1.0, 0.0, 1.0);
	gl_FragColor += float(iInfo.r &   2) * u_overlayOn2 * vec4(1.0, 0.0, 0.0, 1.0);
	gl_FragColor += float(iInfo.r &   4) * u_overlayOn3 * vec4(1.0, 1.0, 0.0, 1.0);
	gl_FragColor += float(iInfo.r &   8) * u_overlayOn4 * vec4(0.0, 1.0, 1.0, 1.0);
	gl_FragColor += float(iInfo.r &  16) * u_overlayOn5 * vec4(0.0, 1.0, 1.0, 1.0);
	gl_FragColor += float(iInfo.r &  32) * u_overlayOn6 * vec4(0.0, 0.0, 0.5, 1.0);
	gl_FragColor += float(iInfo.r &  64) * u_overlayOn7 * vec4(0.0, 1.0, 0.0, 1.0);
	gl_FragColor += float(iInfo.r & 128) * u_overlayOn8 * vec4(1.0, 1.0, 0.0, 1.0);

	gl_FragColor += float(iInfo.g &   1) * u_overlayOn1 * vec4(1.0, 0.0, 1.0, 1.0);
	gl_FragColor += float(iInfo.g &   2) * u_overlayOn2 * vec4(1.0, 1.0, 1.0, 1.0);
	gl_FragColor += float(iInfo.g &   4) * u_overlayOn3 * vec4(1.0, 1.0, 0.0, 1.0);
	gl_FragColor += float(iInfo.g &   8) * u_overlayOn4 * vec4(0.0, 1.0, 1.0, 1.0);
	gl_FragColor += float(iInfo.g &  16) * u_overlayOn5 * vec4(0.0, 1.0, 1.0, 1.0);
	gl_FragColor += float(iInfo.g &  32) * u_overlayOn6 * vec4(0.0, 0.0, 0.5, 1.0);
	gl_FragColor += float(iInfo.g &  64) * u_overlayOn7 * vec4(0.0, 1.0, 0.0, 1.0);
	gl_FragColor += float(iInfo.g & 128) * u_overlayOn8 * vec4(1.0, 0.0, 0.0, 1.0);

	gl_FragColor += float(iInfo.b &   1) * u_overlayOn1 * vec4(1.0, 1.0, 1.0, 1.0);
	gl_FragColor += float(iInfo.b &   2) * u_overlayOn2 * vec4(1.0, 0.0, 0.0, 1.0);
	gl_FragColor += float(iInfo.b &   4) * u_overlayOn3 * vec4(1.0, 1.0, 0.0, 1.0);
	gl_FragColor += float(iInfo.b &   8) * u_overlayOn4 * vec4(0.0, 1.0, 1.0, 1.0);
	gl_FragColor += float(iInfo.b &  16) * u_overlayOn5 * vec4(0.0, 1.0, 1.0, 1.0);
	gl_FragColor += float(iInfo.b &  32) * u_overlayOn6 * vec4(0.0, 0.0, 0.5, 1.0);
	gl_FragColor += float(iInfo.b &  64) * u_overlayOn7 * vec4(0.0, 1.0, 0.0, 1.0);
	gl_FragColor += float(iInfo.b & 128) * u_overlayOn8 * vec4(0.0, 0.0, 1.0, 1.0);
}
//-----------------------------------------------------------------------------
