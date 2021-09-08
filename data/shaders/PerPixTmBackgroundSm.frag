//#############################################################################
//  File:      PerPixTmBackgroundSm.frag
//  Purpose:   GLSL fragment shader for background texture mapping with
//             shadow mapping
//  Date:      November 2020
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces #Lights by SLVLights.size()
#pragma define NUM_LIGHTS #Lights
//-----------------------------------------------------------------------------
in      vec3        v_P_VS;     // Interpol. point of illumination in view space (VS)
in      vec3        v_P_WS;     // Interpol. point of illumination in world space (WS)
in      vec3        v_N_VS;     // Interpol. normal at v_P_VS in view space

uniform bool        u_lightIsOn[NUM_LIGHTS];                // flag if light is on
uniform vec4        u_lightPosWS[NUM_LIGHTS];               // position of light in world space
uniform vec4        u_lightPosVS[NUM_LIGHTS];               // position of light in world space
uniform vec3        u_lightSpotDir[NUM_LIGHTS];             // spot direction in view space
uniform bool        u_lightCreatesShadows[NUM_LIGHTS];      // flag if light creates shadows
uniform bool        u_lightDoSmoothShadows[NUM_LIGHTS];     // flag if percentage-closer filtering is enabled
uniform int         u_lightSmoothShadowLevel[NUM_LIGHTS];   // radius of area to sample for PCF
uniform float       u_lightShadowMinBias[NUM_LIGHTS];       // min. shadow bias value at 0° to N
uniform float       u_lightShadowMaxBias[NUM_LIGHTS];       // min. shadow bias value at 90° to N

uniform sampler2D   u_shadowMap_0;      // shadow map for light 0
uniform mat4        u_lightSpace_0;     // projection matrices for lights

uniform float       u_camBkgdWidth;     // background width
uniform float       u_camBkgdHeight;    // background height
uniform float       u_camBkgdLeft;      // background left
uniform float       u_camBkgdBottom;    // background bottom

uniform bool        u_matGetsShadows;   // flag if material receives shadows
uniform vec4        u_matAmbi;          // ambient color reflection coefficient (ka)

uniform sampler2D   u_matTexture0;      // Color map

out     vec4        o_fragColor;        // output fragment color
//-----------------------------------------------------------------------------
#pragma include "shadowTest1Light.glsl"
//-----------------------------------------------------------------------------
void main()
{
    float x = (gl_FragCoord.x - u_camBkgdLeft) / u_camBkgdWidth;
    float y = (gl_FragCoord.y - u_camBkgdBottom) / u_camBkgdHeight;

    vec4 fragVideo;
    if(x < 0.0f || y < 0.0f || x > 1.0f || y > 1.0f)
        fragVideo = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    else
        fragVideo = texture(u_matTexture0, vec2(x, y));
        
    vec3 N = normalize(v_N_VS);  // A input normal has not anymore unit length
    float shadow = 0.0;

    if (u_lightIsOn[0])
    {
        if (u_lightPosVS[0].w == 0.0)
        {
            // We use the spot light direction as the light direction vector
            vec3 S = normalize(-u_lightSpotDir[0].xyz);

            // Test if the current fragment is in shadow
            shadow = u_matGetsShadows ? shadowTest1Light(0, N, S) : 0.0;
        }
        else
        {
            vec3 L = u_lightPosVS[0].xyz - v_P_VS; // Vector from v_P to light in VS

            // Test if the current fragment is in shadow
            shadow = u_matGetsShadows ? shadowTest1Light(0, N, L) : 0.0;
        }
    }

    o_fragColor = fragVideo * min(1.0 - shadow + u_matAmbi.r, 1.0);
}
//-----------------------------------------------------------------------------
