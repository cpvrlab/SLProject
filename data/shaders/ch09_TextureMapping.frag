//#############################################################################
//  File:      ch09_TextureMapping.frag
//  Purpose:   GLSL fragment shader for per pixel Blinn-Phong lighting w. tex.
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec3        v_P_VS;          // Interpol. point of illumination in view space
in      vec3        v_N_VS;          // Interpol. normal at v_P_VS in view space
in      vec2        v_uv;            // Interpol. texture coordinate

uniform vec4        u_lightPosVS;    // position of light in view space
uniform vec4        u_lightAmbi;     // ambient light intensity (Ia)
uniform vec4        u_lightDiff;     // diffuse light intensity (Id)
uniform vec4        u_lightSpec;     // specular light intensity (Is)
uniform vec3        u_lightSpotDir;  // spot direction in view space
uniform float       u_lightSpotDeg;  // spot cutoff angle 1-180 degrees
uniform float       u_lightSpotCos;  // cosine of spot cutoff angle
uniform float       u_lightSpotExp;  // spot exponent
uniform vec3        u_lightAtt;      // attenuation (const,linear,quadr.)
uniform vec4        u_globalAmbi;    // Global ambient scene color

uniform vec4        u_matAmbi;       // ambient color reflection coefficient (ka)
uniform vec4        u_matDiff;       // diffuse color reflection coefficient (kd)
uniform vec4        u_matSpec;       // specular color reflection coefficient (ks)
uniform vec4        u_matEmis;       // emissive color for self-shining materials
uniform float       u_matShin;       // shininess exponent
uniform sampler2D   u_matTexDiff;    // diffuse color texture map

out     vec4   o_fragColor;     // output fragment color
//-----------------------------------------------------------------------------
void pointLightBlinnPhong(in    vec3  N,  // Normalized normal at v_P
                          in    vec3  E,  // Normalized direction at v_P to the eye
                          in    vec3  S,  // Normalized light spot direction
                          in    vec3  L,  // Unnormalized direction at v_P to the light
                          inout vec4  Ia, // Ambient light intensity
                          inout vec4  Id, // Diffuse light intensity
                          inout vec4  Is) // Specular light intensity
{
    // Light attenuation
    // By default, there is no attenuation set. This is physically not correct
    // Default OpenGL:      kc=1, kl=0, kq=0
    // Physically correct:  kc=0, kl=0, kq=1
    // set quadratic attenuation with d = distance to light
    //               1
    // att = ------------------
    //       kc + kl*d + kq*d*d
    vec3 att_dist;
    att_dist.x = 1.0;
    att_dist.z = dot(L, L);// = distance * distance
    att_dist.y = sqrt(att_dist.z);// = distance
    float att = 1.0 / dot(att_dist, u_lightAtt);
    L /= att_dist.y; // = normalize(L)

    // Calculate diffuse & specular factors
    vec3 H = normalize(E + L); // Blinn's half vector is faster than Phongs reflected vector
    float diffFactor = max(dot(N, L), 0.0); // Lambertian downscale factor for diffuse reflection
    float specFactor = 0.0;
    if (diffFactor!=0.0) // specular reflection is only possible if surface is lit from front
    specFactor = pow(max(dot(N, H), 0.0), u_matShin); // specular shininess

    // Calculate spoteffect & spot attenuation
    if (u_lightSpotDeg < 180.0)
    {
        float spotAtt = 1.0; // Spot attenuation
        // ???
        att *= spotAtt;
    }

    // Accumulate light intesities except the ambient part
    Ia += u_lightAmbi;
    Id += att * u_lightDiff * diffFactor;
    Is += att * u_lightSpec * specFactor;
}
//-----------------------------------------------------------------------------
void main()
{
    vec3 N = normalize(v_N_VS);// A input normal has not anymore unit length
    vec3 E = normalize(-v_P_VS);// Vector from p to the eye
    vec3 S = u_lightSpotDir; // normalized spot direction in VS
    vec3 L = u_lightPosVS.xyz - v_P_VS; // Vector from v_P to light in VS

    vec4 Ia = vec4(0.0); // Accumulated ambient light intensity at v_P_VS
    vec4 Id = vec4(0.0); // Accumulated diffuse light intensity at v_P_VS
    vec4 Is = vec4(0.0); // Accumulated specular light intensity at v_P_VS

    pointLightBlinnPhong(N, E, S, L, Ia, Id, Is);

    // Sum up all the reflected color components except the specular part
    // The specular part would be deleted by the component-wise multiplication
    // of the texture.
    o_fragColor = u_matEmis +
                  u_globalAmbi +
                  Ia * u_matAmbi +
                  Id * u_matDiff;

    // Componentwise multiply w. texture color
    o_fragColor *= texture(u_matTexDiff, v_uv);

    // Add finally the specular RGB-part
    vec4 specColor = Is * u_matSpec;
    o_fragColor.rgb += specColor.rgb;
}
//-----------------------------------------------------------------------------

