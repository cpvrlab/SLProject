//#############################################################################
//  File:      PerVrtBlinnTex.frag
//  Purpose:   GLSL per vertex Blinn-Phong lighting without texturing
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
in      vec4        v_color;                // Interpol. ambient & diff. color
in      vec4        v_specColor;            // Interpol. specular color
in      vec2        v_texCoord;             // Interpol. texture coordinate

uniform sampler2D   u_texture0;             // Color map
uniform int         u_projection;           // type of stereo
uniform int         u_stereoEye;            // -1=left, 0=center, 1=right
uniform mat3        u_stereoColorFilter;    // color filter matrix
uniform float       u_oneOverGamma;         // 1.0f / Gamma correction value

out     vec4        o_fragColor;            // output fragment color
//-----------------------------------------------------------------------------
void main()
{
    // Interpolated ambient and diffuse components  
    o_fragColor = v_color;
   
    // componentwise multiply w. texture color
    o_fragColor *= texture(u_texture0, v_texCoord);
   
    // add finally the specular RGB part but not alpha
    o_fragColor.rgb += v_specColor.rgb;
   
    // Apply stereo eye separation
    if (u_projection > 1)
    {   if (u_projection > 7) // stereoColor??
        {   // Apply color filter but keep alpha
            o_fragColor.rgb = u_stereoColorFilter * o_fragColor.rgb;
        }
        else if (u_projection == 5) // stereoLineByLine
        {   if (mod(floor(gl_FragCoord.y), 2.0) < 0.5) // even
            {   if (u_stereoEye ==-1) discard;
            } else // odd
            {   if (u_stereoEye == 1) discard;
            }
        }
        else if (u_projection == 6) // stereoColByCol
        {   if (mod(floor(gl_FragCoord.x), 2.0) < 0.5) // even
            {   if (u_stereoEye ==-1) discard;
            } else // odd
            {   if (u_stereoEye == 1) discard;
            }
        } 
        else if (u_projection == 7) // stereoCheckerBoard
        {   bool h = (mod(floor(gl_FragCoord.x), 2.0) < 0.5);
            bool v = (mod(floor(gl_FragCoord.y), 2.0) < 0.5);
            if (h==v) // both even or odd
            {   if (u_stereoEye ==-1) discard;
            } else // odd
            {   if (u_stereoEye == 1) discard;
            }
        }
    }

    // Apply gamma correction on diffuse part
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
//-----------------------------------------------------------------------------
