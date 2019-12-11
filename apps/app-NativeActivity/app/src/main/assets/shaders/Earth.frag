//#############################################################################
//  File:      BumpParallax.frag
//  Purpose:   OGLSL parallax bump mapping
//  Author:    Marcus Hudritsch
//  Date:      18-SEP-09 (HS09)
//  Copyright: Marcus Hudritsch
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

uniform sampler2D Texture0;   // Color map
uniform sampler2D Texture1;   // Normal map
uniform sampler2D Texture2;   // Height map;
uniform sampler2D Texture3;   // Gloss map;
uniform float     scale;      // Height scale factor for parallax mapping
uniform float     bias;       // Height bias for parallax mapping
varying vec3      L_TS;       // Vector to the light in tangent space
varying vec3      E_TS;       // Vector to the eye in tangent space
varying vec3      S_TS;       // Spot direction in tangent space
varying float     d;          // Light distance

void main()
{
   // Normalize E and L
   vec3 E = normalize(E_TS);
   vec3 L = normalize(L_TS);
   
   // Halfvector H between L & E (See Blinn's lighting model)
   vec3 H = normalize(L + E);
   
   ////////////////////////////////////////////////////////////
   // Calculate new texture coord. Tc for Parallax mapping
   // The height comes from red channel from the height map
   float height = texture2D(Texture2, gl_TexCoord[0].st).r;
   
   // Scale the height and add the bias (height offset)
   height = height * scale + bias;
   
   // Add the texture offset to the texture coord.
   vec2 Tc = gl_TexCoord[0].st + (height * E.st);
   ////////////////////////////////////////////////////////////
   
   // Get normal from normal map, move from [0,1] to [-1, 1] range & normalize
   vec3 N = normalize(texture2D(Texture1, Tc).rgb * 2.0 - 1.0);
   
   // Calculate attenuation over distance d
   float att = 1.0 / (gl_LightSource[0].constantAttenuation +	
                      gl_LightSource[0].linearAttenuation * d +	
                      gl_LightSource[0].quadraticAttenuation * d * d);
   
   // Calculate spot attenuation
   if (gl_LightSource[0].spotCutoff < 180.0)
   {  float spotDot; // Cosine of angle between L and spotdir
      float spotAtt; // Spot attenuation
      vec3 S = normalize(S_TS);
      spotDot = dot(-L, S);
      if (spotDot < gl_LightSource[0].spotCosCutoff) spotAtt = 0.0;
      else spotAtt = pow(spotDot, gl_LightSource[0].spotExponent);
      att *= spotAtt;
   }
   
   // compute diffuse lighting
   float diffFactor = max(dot(L,N), 0.0) ;
   
   // compute specular lighting
   float specFactor = pow(max(dot(N,H), 0.0), gl_FrontMaterial.shininess * 10.0);
   
   // add ambient & diffuse light components
   gl_FragColor = gl_FrontLightModelProduct.sceneColor;
   gl_FragColor += att * gl_LightSource[0].ambient * gl_FrontMaterial.ambient;
   gl_FragColor += att * gl_LightSource[0].diffuse * gl_FrontMaterial.diffuse * diffFactor;
   
   // componentwise multiply w. texture color (= GL_MODULATE)
   gl_FragColor *= texture2D(Texture0, Tc);
   
   // add finally the specular part
   gl_FragColor += att * 
                   gl_LightSource[0].specular * 
                   gl_FrontMaterial.specular * 
                   specFactor;
}
