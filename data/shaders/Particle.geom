//#############################################################################
//  File:      Particle.geom
//  Purpose:   GLSL geom program for particle system
//  Date:      October 2021
//  Authors:   Marc Affolter
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------

layout (triangles) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;    

in Vertex
{
	vec4 vv_particleColor; // The resulting color per vertex
	vec2 vv_texCoord;
} vertex[];

out vec2 v_texCoord; // texture coordinate at vertex
out vec4 v_particleColor; // The resulting color per vertex
//-----------------------------------------------------------------------------
void main (void)
{
  gl_Position = gl_in[0].gl_Position;
  v_texCoord = vertex[0].vv_texCoord;
  v_particleColor = vertex[0].vv_particleColor;
  EmitVertex();

  gl_Position = gl_in[1].gl_Position;
  v_texCoord = vertex[1].vv_texCoord;
  v_particleColor = vertex[1].vv_particleColor;
  EmitVertex();

  gl_Position = gl_in[2].gl_Position;
  v_texCoord = vertex[2].vv_texCoord;
  v_particleColor = vertex[2].vv_particleColor;
  EmitVertex();
  
  EndPrimitive();  
}   