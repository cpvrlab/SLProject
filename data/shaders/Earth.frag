//#############################################################################
//  File:      BumpParallax.frag
//  Purpose:   OGLSL parallax bump mapping
//  Date:      18-SEP-09 (HS09)
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in       vec3      L_TS;        // Vector to the light in tangent space
in       vec3      E_TS;        // Vector to the eye in tangent space
in       vec3      S_TS;        // Spot direction in tangent space
in       float     d;           // Light distance

uniform  sampler2D TextureDiffuse0;   // Color map
uniform  sampler2D TextureNormal0;    // Normal map
uniform  sampler2D TextureHeight0;    // Height map;
uniform  sampler2D TextureSpecular0;  // Gloss map;
uniform  float     scale;       // Height scale factor for parallax mapping
uniform  float     bias;        // Height bias for parallax mapping

out      vec4      o_fragColor; // output fragment color
//-----------------------------------------------------------------------------
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
   float height = texture(TextureHeight0, gl_TexCoord[0].st).r;
   
   // Scale the height and add the bias (height offset)
   height = height * scale + bias;
   
   // Add the texture offset to the texture coord.
   vec2 Tc = gl_TexCoord[0].st + (height * E.st);
   ////////////////////////////////////////////////////////////
   
   // Get normal from normal map, move from [0,1] to [-1, 1] range & normalize
   vec3 N = normalize(texture(TextureNormal0, Tc).rgb * 2.0 - 1.0);
   
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
   o_fragColor = gl_FrontLightModelProduct.sceneColor;
   o_fragColor += att * gl_LightSource[0].ambient * gl_FrontMaterial.ambient;
   o_fragColor += att * gl_LightSource[0].diffuse * gl_FrontMaterial.diffuse * diffFactor;
   
   // componentwise multiply w. texture color (= GL_MODULATE)
   o_fragColor *= texture(TextureDiffuse0, Tc);
   
   // add finally the specular part
   o_fragColor += att *
                   gl_LightSource[0].specular * 
                   gl_FrontMaterial.specular * 
                   specFactor;
}
//-----------------------------------------------------------------------------
