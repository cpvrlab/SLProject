//#############################################################################
//  File:      ShadowMapping.frag
//  Author:    Micha Stettler
//  Date:      July 2014
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision mediump float;
#endif

//-----------------------------------------------------------------------------
in      vec3            P_VS;
in      vec3            N_VS;
in      vec4            ShadowCoord0;
in      vec4            ShadowCoord1;
in      vec4            ShadowCoord2;
in      vec4            ShadowCoord3;
in      vec4            ShadowCoord4;
in      vec4            ShadowCoord5;
in      vec4            ShadowCoord6;
in      vec4            ShadowCoord7;

uniform sampler2DShadow ShadowMap0;
uniform sampler2DShadow ShadowMap1;
uniform sampler2DShadow ShadowMap2;
uniform sampler2DShadow ShadowMap3;
uniform sampler2DShadow ShadowMap4;
uniform sampler2DShadow ShadowMap5;
uniform sampler2DShadow ShadowMap6;
uniform sampler2DShadow ShadowMap7;
uniform int             numLights;
uniform float           xPixelOffset;
uniform float           yPixelOffset;

out     vec4        o_fragColor;    // output fragment color
//-----------------------------------------------------------------------------
vec4 Ia = vec4(0.0);  //Ambient
vec4 Id = vec4(0.0);  //Diffuse
vec4 Is = vec4(0.0);  //Specular
//-----------------------------------------------------------------------------
float lookup( vec2 offSet, vec4 ShadowCoord, sampler2DShadow ShadowMap)
{
    // Returns 0 if the Point is in Shadow and 1 if its not (CompareMode: GL_EQUAL)
    return shadow2DProj(ShadowMap, ShadowCoord + vec4(offSet.x * xPixelOffset * ShadowCoord.w, 
                                                      offSet.y * yPixelOffset * ShadowCoord.w,
                                                      0.005, 
                                                      0.0)).w;
}
//-----------------------------------------------------------------------------
void shadeBlinnPhongWithShadows(const int i)
{
    ///////////////////
    // PCF Filtering //
    ///////////////////

    // a x b Filter Matrix (1x1 = no Filtering, 2x2 = standard, 4x4 better Quality but huge FPS loss)
    float a = 2.0;
    float b = 2.0;

    float x,y;

    //Calculate the min and max values, used in the Filter Loop
    float min_a = (a/2.0)-0.5;
    float max_a = (a/2.0)-0.5;
    float min_b = (b/2.0)-0.5;
    float max_b = (b/2.0)-0.5;

    //Filter Loop
    float shadow = 0.0;
    for (y = -min_a; y <=max_a ; y+= 1.0)
    {   for (x = -min_b ; x <=max_b ; x+=1.0)
        {   //lookup Shadow in the matching ShadowMap
            if(i == 0 && ShadowCoord0.w > 1.0)
                shadow += lookup(vec2(x,y),ShadowCoord0,ShadowMap0);
            else if(i == 1 && ShadowCoord1.w > 1.0)
                shadow += lookup(vec2(x,y),ShadowCoord1,ShadowMap1);
            else if(i == 2 && ShadowCoord2.w > 1.0)
                shadow += lookup(vec2(x,y),ShadowCoord2,ShadowMap2);
            else if(i == 3 && ShadowCoord3.w > 1.0)
                shadow += lookup(vec2(x,y),ShadowCoord3,ShadowMap3);
            else if(i == 4 && ShadowCoord4.w > 1.0)
                shadow += lookup(vec2(x,y),ShadowCoord4,ShadowMap4);
            else if(i == 5 && ShadowCoord5.w > 1.0)
                shadow += lookup(vec2(x,y),ShadowCoord5,ShadowMap5);
            else if(i == 6 && ShadowCoord6.w > 1.0)
                shadow += lookup(vec2(x,y),ShadowCoord6,ShadowMap6);
            else if(i == 7 && ShadowCoord7.w > 1.0)
                shadow += lookup(vec2(x,y),ShadowCoord7,ShadowMap7);
        }
    }

    shadow /= (a*b);

    /////////////////////////
    // Blinn Phong Shading //
    /////////////////////////

    // Vector to the Light
    vec3 L = vec3(u_light[i].position.xyz - P_VS);

    //Length of the Vector from Position to Light
    float d = length(L);

    //Normalize Lightvector
    L = normalize(L);

    vec3 N = normalize(N_VS);
    vec3 E = normalize(-P_VS);
    vec3 H = normalize(L + E);

    // Calculate diffuse & specular factors
    float nDotL = max(dot(N,L), 0.0);
    float shine;
    if (nDotL==0.0) shine = 0.0;
    else shine = pow(max(dot(N,H), 0.0), u_matShin);
   
    // Calculate attenuation over distance d
    float att = 1.0 / (u_light[i].constantAttenuation +	
                       u_light[i].linearAttenuation * d +	
                       u_light[i].quadraticAttenuation * d * d);

    // Calculate spot attenuation
    if (u_lightSpotCos[i] < 180.0)
    {   float spotDot; // Cosine of angle between L and spotdir
        float spotAtt; // Spot attenuation
        spotDot = dot(-L, u_light[i].spotDirection);
        if (spotDot < u_lightSpotCos[i]) spotAtt = 0.0;
        else spotAtt = pow(spotDot, u_lightSpotExp[i]);
        att *= spotAtt;
    }

    /////////////////////////
    // Combination         //
    /////////////////////////

    // Accumulate light intesities and shadow
    shadow = shadow * nDotL;

    Ia += u_lightAmbi[i]  * att * shadow ;
    Id += u_lightDiff[i]  * att * nDotL * shadow;
    Is += u_lightSpec[i] * att * shine *shadow;
}
//-----------------------------------------------------------------------------
void main()
{
    //Go through each active Light and do the lightning / shadowing
    for(int i=0; i < numLights; i++)
        if (u_light[i].position.w >= 0.0)
            shadeBlinnPhongWithShadows(i);

    //Update the Fragment Color
    o_fragColor =  u_sceneColor +
                    Ia * u_matAmbi +
                    Id * u_matDiff;

    o_fragColor += Is * u_matSpec;
}
//-----------------------------------------------------------------------------
