//#############################################################################
//  File:      PBR_BRDFIntegration.vert
//  Purpose:   GLSL vertex program for generating a BRDF integration map, which
//             is the second part of the specular integral.
//  Author:    Carlos Arauz
//  Date:      April 2018
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
layout (location = 0) in vec4  a_position;  // Vertex position attribute
layout (location = 2) in vec2  a_uv1;       // Vertex texture attribute

out                      vec2  v_uv1;       // texture coordinate at vertex
//-----------------------------------------------------------------------------
void main()
{
    v_uv1  = a_uv1;
    gl_Position = vec4(a_position, 1.0);
}
//-----------------------------------------------------------------------------
