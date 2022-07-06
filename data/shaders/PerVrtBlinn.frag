//#############################################################################
//  File:      PerVrtBlinn.frag
//  Purpose:   GLSL per vertex Blinn-Phong lighting without texturing
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in      vec3    v_P_VS;             // Interpol. point of illumination in view space (VS)
in      vec4    v_color;            // interpolated color from vertex shader

uniform int     u_camProjType;    // type of stereo
uniform int     u_camStereoEye;     // -1=left, 0=center, 1=right
uniform mat3    u_camStereoColors;  // color filter matrix
uniform bool    u_camFogIsOn;       // flag if fog is on
uniform int     u_camFogMode;       // 0=LINEAR, 1=EXP, 2=EXP2
uniform float   u_camFogDensity;    // fog density value
uniform float   u_camFogStart;      // fog start distance
uniform float   u_camFogEnd;        // fog end distance
uniform vec4    u_camFogColor;      // fog color (usually the background)

out     vec4    o_fragColor;        // output fragment color
//-----------------------------------------------------------------------------
// SLGLShader::preprocessPragmas replaces the include pragma by the file
#pragma include "fogBlend.glsl"
#pragma include "doStereoSeparation.glsl
//-----------------------------------------------------------------------------
void main()
{     
    o_fragColor = v_color;

    // Apply fog by blending over distance
    if (u_camFogIsOn)
        o_fragColor = fogBlend(v_P_VS, o_fragColor);
   
    // Apply stereo eye separation
    if (u_camProjType > 1)
        doStereoSeparation();
}
//-----------------------------------------------------------------------------
