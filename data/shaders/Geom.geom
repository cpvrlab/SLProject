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

out vec4 v_particleColor; // The resulting color per vertex
out vec2 v_texCoord; 
//-----------------------------------------------------------------------------
void main (void)
{
  vec3 va = vec3(-0.2, -0.2, 0);
  va += gl_in[0].gl_Position.xyz;
  gl_Position =  vec4(va, 1.0);
  v_texCoord = vec2(0.0, 0.0);
  v_particleColor = vec4(1.0, 0.0, 0.0, 1.0); // red 
  EmitVertex();  
  
  vec3 vb = vec3(-0.2, 0.2, 0);
  vb += gl_in[0].gl_Position.xyz;
  gl_Position = vec4(vb, 1.0);
  v_texCoord = vec2(0.0, 1.0);
  v_particleColor = vec4(0.0, 1.0, 0.0, 1.0); // green
  EmitVertex();  

  vec3 vd = vec3(0.2, -0.2, 0);
  vd += gl_in[0].gl_Position.xyz;
  gl_Position = vec4(vd, 1.0);
  v_texCoord = vec2(1.0, 0.0);
  v_particleColor = vec4(0.0, 0.0, 1.0, 1.0); // blue
  EmitVertex();  

  vec3 vc = vec3(0.2, 0.2, 0);
  vc += gl_in[0].gl_Position.xyz;
  gl_Position = vec4(vc, 1.0);
  v_texCoord = vec2(1.0, 1.0);
  v_particleColor = vec4(1.0, 1.0, 1.0, 1.0);
  EmitVertex();  
  
  EndPrimitive();  
}   