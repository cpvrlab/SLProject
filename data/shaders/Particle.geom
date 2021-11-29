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
    float transparency;
    vec4 pos;
} vert[];

uniform mat4 u_pMatrix;  // projection matrix

uniform vec4 u_color; // Object color
uniform float u_scale; // Object scale

out vec4 v_particleColor; // The resulting color per vertex
out vec2 v_texCoord; //Texture coordinate at vertex
//-----------------------------------------------------------------------------
void main (void)
{

  vec4 P = gl_in[0].gl_Position;

  //BOTTOM LEFT
  vec4 va = vec4((vert[0].pos.xy + vec2(-0.5, -0.5)) ,vert[0].pos.z,0) + P;
  gl_Position = u_pMatrix * va;
  v_texCoord = vec2(0.0, 0.0);
  v_particleColor = u_color;
  v_particleColor.w = vert[0].transparency;
  EmitVertex();  
  
  //BOTTOM RIGHT
  vec4 vd = vec4((vert[0].pos.xy + vec2(0.5, -0.5)) * u_scale,vert[0].pos.z,0) + P;
  gl_Position = u_pMatrix * vd;
  v_texCoord = vec2(1.0, 0.0);
  v_particleColor = u_color;
  v_particleColor.w = vert[0].transparency;
  EmitVertex();  

  //TOP LEFT
  vec4 vb = vec4((vert[0].pos.xy + vec2(-0.5, 0.5)) * u_scale,vert[0].pos.z,0) + P;
  gl_Position = u_pMatrix * vb;
  v_texCoord = vec2(0.0, 1.0);
  v_particleColor = u_color;
  v_particleColor.w = vert[0].transparency;
  EmitVertex();  

  //TOP RIGHT
  vec4 vc = vec4((vert[0].pos.xy + vec2(0.5, 0.5)) * u_scale,vert[0].pos.z,0) + P;
  gl_Position = u_pMatrix *  vc;
  v_texCoord = vec2(1.0, 1.0);
  v_particleColor = u_color;
  v_particleColor.w = vert[0].transparency;
  EmitVertex();  
  
  EndPrimitive();  
}   