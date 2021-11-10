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

layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;

in vertex {
    vec3 color;
} vertex[];

uniform mat4 u_pMatrix;  // projection matrix

uniform vec4 u_color; // Object color
uniform float u_scale; // Object scale
uniform vec3 u_offset;	// Object offset

out vec4 v_particleColor; // The resulting color per vertex
out vec2 v_texCoord; //Texture coordinate at vertex
//-----------------------------------------------------------------------------
void main (void)
{
  vec4 P = gl_in[0].gl_Position;

  //BOTTOM LEFT
  vec2 va = (P.xy + vec2(-0.5, -0.5)) * u_scale + u_offset.xy;
  gl_Position = u_pMatrix * vec4(va, P.z + u_offset.z, P.w);
  v_texCoord = vec2(0.0, 0.0);
  v_particleColor = u_color;
  v_particleColor.w = vertex[0];
  EmitVertex();  
  
  //BOTTOM RIGHT
  vec2 vd = (P.xy + vec2(0.5, -0.5)) * u_scale + u_offset.xy;
  gl_Position = u_pMatrix *  vec4(vd, P.z + u_offset.z, P.w);
  v_texCoord = vec2(1.0, 0.0);
  v_particleColor = u_color;
  v_particleColor.w = vertex[0];
  EmitVertex();  

  //TOP LEFT
  vec2 vb = (P.xy + vec2(-0.5, 0.5)) * u_scale + u_offset.xy;
  gl_Position = u_pMatrix * vec4(vb, P.z + u_offset.z, P.w);
  v_texCoord = vec2(0.0, 1.0);
  v_particleColor = u_color;
  v_particleColor.w = vertex[0];
  EmitVertex();  

  //TOP RIGHT
  vec2 vc = (P.xy + vec2(0.5, 0.5)) * u_scale + u_offset.xy;
  gl_Position = u_pMatrix *  vec4(vc, P.z + u_offset.z, P.w);
  v_texCoord = vec2(1.0, 1.0);
  v_particleColor = u_color;
  v_particleColor.w = vertex[0];
  EmitVertex();  
  
  EndPrimitive();  
}   