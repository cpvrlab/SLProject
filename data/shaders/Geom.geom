//#############################################################################
//  File:      Geom.geom
//  Purpose:   GLSL geom program for geom exercise
//  Date:      November 2021
//  Authors:   Marc Affolter
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------

layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;

uniform mat4 u_pMatrix;     // projection matrix

uniform vec4 u_color;       // Object color
uniform float u_scale;      // Object scale

out vec4 v_particleColor;   // The resulting color per vertex
out vec2 v_texCoord;        //Texture coordinate at vertex
//-----------------------------------------------------------------------------
void main (void)
{
  vec4 P = gl_in[0].gl_Position;

  //BOTTOM LEFT
  vec2 va = (P.xy + vec2(-0.5, -0.5)) * u_scale;
  gl_Position = u_pMatrix * vec4(va, P.zw);
  v_texCoord = vec2(0.0, 0.0);
  v_particleColor = u_color;
  EmitVertex();  
  
  //BOTTOM RIGHT
  vec2 vd = (P.xy + vec2(0.5, -0.5)) * u_scale;
  gl_Position = u_pMatrix *  vec4(vd, P.zw);
  v_texCoord = vec2(1.0, 0.0);
  v_particleColor = u_color;
  EmitVertex();  

  //TOP LEFT
  vec2 vb = (P.xy + vec2(-0.5, 0.5)) * u_scale;
  gl_Position = u_pMatrix * vec4(vb, P.zw);
  v_texCoord = vec2(0.0, 1.0);
  v_particleColor = u_color;
  EmitVertex();  

  //TOP RIGHT
  vec2 vc = (P.xy + vec2(0.5, 0.5)) * u_scale;
  gl_Position = u_pMatrix *  vec4(vc, P.zw);
  v_texCoord = vec2(1.0, 1.0);
  v_particleColor = u_color;
  EmitVertex();  
  
  EndPrimitive();  
}   