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

uniform mat4 u_mvpMatrix;  // modelview-projection matrix = projection * modelView
uniform mat4 u_mvMatrix;  // modelview matrix

uniform vec4 color;
uniform vec3 offset; 
uniform float scale; // Particle scale 

out vec2 v_texCoord; // texture coordinate at vertex
out vec4 v_particleColor; // The resulting color per vertex
//-----------------------------------------------------------------------------
void main (void)
{
  vec3 va = vec3(-0.5, -0.5, 0);
  gl_Position = u_mvpMatrix * vec4(va, 1.0);
  v_texCoord = vec2(0.0, 0.0);
  v_particleColor = color;
  EmitVertex();  
  
  vec3 vb = vec3(-0.5, 0.5, 0);
  gl_Position = u_mvpMatrix * vec4(vb, 1.0);
  v_texCoord = vec2(0.0, 1.0);
  v_particleColor = color;
  EmitVertex();  

  vec3 vd = vec3(0.5, -0.5, 0);
  gl_Position = u_mvpMatrix * vec4(vd, 1.0);
  v_texCoord = vec2(1.0, 0.0);
  v_particleColor = color;
  EmitVertex();  

  vec3 vc = vec3(0.5, 0.5, 0);
  gl_Position = u_mvpMatrix * vec4(vc, 1.0);
  v_texCoord = vec2(1.0, 1.0);
  v_particleColor = color;
  EmitVertex();  
  
  EndPrimitive();  
}   