//#############################################################################
//  File:      PerVrtTextureOnly.frag
//  Purpose:   GLSL fragment shader for texture mapping only
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_ES
precision mediump float;
#endif
//-----------------------------------------------------------------------------
in      vec3        v_P_VS;           // Interpol. point of illum. in view space (VS)
in      vec2        v_texCoord;       // Interpol. texture coordinate

uniform sampler2D   u_matTexture0;    // Color map
uniform float       u_oneOverGamma;   // 1.0f / Gamma correction value

uniform int         u_camProjection;    // type of stereo
uniform int         u_camStereoEye;     // -1=left, 0=center, 1=right
uniform mat3        u_camStereoColors;  // color filter matrix
uniform bool        u_camFogIsOn;       // flag if fog is on
uniform int         u_camFogMode;       // 0=LINEAR, 1=EXP, 2=EXP2
uniform float       u_camFogDensity;    // fog densitiy value
uniform float       u_camFogStart;      // fog start distance
uniform float       u_camFogEnd;        // fog end distance
uniform vec4        u_camFogColor;      // fog color (usually the background)

out     vec4        o_fragColor;      // output fragment color
//-----------------------------------------------------------------------------
vec4 fogBlend(vec3 P_VS, vec4 inColor)
{
    float factor = 0.0f;
    float distance = length(P_VS);

    switch (u_camFogMode)
    {
        case 0:
            factor = (u_camFogEnd - distance) / (u_camFogEnd - u_camFogStart);
            break;
        case 1:
            factor = exp(-u_camFogDensity * distance);
            break;
        default:
            factor = exp(-u_camFogDensity * distance * u_camFogDensity * distance);
            break;
    }

    vec4 outColor = factor * inColor + (1 - factor) * u_camFogColor;
    outColor = clamp(outColor, 0.0, 1.0);
    return outColor;
}
//-----------------------------------------------------------------------------
void main()
{     
    o_fragColor = texture(u_matTexture0, v_texCoord);

    // Apply fog by blending over distance
    if (u_camFogIsOn)
        o_fragColor = fogBlend(v_P_VS, o_fragColor);

    // Apply stereo eye separation
    if (u_camProjection > 1)
    {   if (u_camProjection > 7) // stereoColor??
        {   // Apply color filter but keep alpha
            o_fragColor.rgb = u_camStereoColors * o_fragColor.rgb;
        }
        else if (u_camProjection == 5) // stereoLineByLine
            {   if (mod(floor(gl_FragCoord.y), 2.0) < 0.5) // even
    {   if (u_camStereoEye ==-1) discard;
    } else // odd
    {   if (u_camStereoEye == 1) discard;
    }
    }
    else if (u_camProjection == 6) // stereoColByCol
    {   if (mod(floor(gl_FragCoord.x), 2.0) < 0.5) // even
    {   if (u_camStereoEye ==-1) discard;
    } else // odd
    {   if (u_camStereoEye == 1) discard;
    }
    }
    else if (u_camProjection == 7) // stereoCheckerBoard
    {   bool h = (mod(floor(gl_FragCoord.x), 2.0) < 0.5);
        bool v = (mod(floor(gl_FragCoord.y), 2.0) < 0.5);
        if (h==v) // both even or odd
        {   if (u_camStereoEye ==-1) discard;
        } else // odd
        {   if (u_camStereoEye == 1) discard;
        }
    }
    }

    // Apply gamma correction on diffuse part
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
