//#############################################################################
//  File:      PBR_BRDFIntegration.vert
//  Purpose:   GLSL vertex program for generating a BRDF integration map, which
//             is the second part of the specular integral.
//  Date:      April 2018
//  Authors:   Carlos Arauz, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in  vec3   a_position; // Vertex position attribute
in  vec2   a_uv0;      // Vertex texture coord. attribute

out vec2   v_uv0;      // Output for interpolated texture coord.
//-----------------------------------------------------------------------------
void main()
{
    v_uv0  = a_uv0;
    gl_Position = vec4(a_position, 1.0);
}
//-----------------------------------------------------------------------------
