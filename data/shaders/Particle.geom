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

layout (points) in;             // Primitives that we received from vertex shader
layout (triangle_strip) out;    // Primitives that we will output
layout (max_vertices = 4) out;  // Number of vertex that will be output

in vertex {
    float transparency; // Transparency of a particle
} vert[];

uniform mat4 u_pMatrix; // Projection matrix

uniform vec4 u_color;   // Particle color
uniform float u_scale;  // Particle scale
uniform float u_radius; // Particle radius

out vec4 v_particleColor;   // The resulting color per vertex
out vec2 v_texCoord;        // Texture coordinate at vertex
//-----------------------------------------------------------------------------
void main (void)
{

  vec4 P = gl_in[0].gl_Position;    // Position of the point that we received

  // Create 4 points to create two triangle to draw one particle

  //BOTTOM LEFT
  vec4 va = vec4((P.xy + vec2(-u_radius, -u_radius)) * u_scale , P.z * u_scale, 1);
  gl_Position = u_pMatrix * va; // Calculate position in clip space
  v_texCoord = vec2(0.0, 0.0);  // Posi
  v_particleColor = u_color;
  v_particleColor.w = vert[0].transparency;
  EmitVertex();  
  
  //BOTTOM RIGHT
  vec4 vd = vec4((P.xy + vec2(u_radius, -u_radius)) * u_scale, P.z * u_scale,1);
  gl_Position = u_pMatrix * vd;
  v_texCoord = vec2(1.0, 0.0);
  v_particleColor = u_color;
  v_particleColor.w = vert[0].transparency;
  EmitVertex();  

  //TOP LEFT
  vec4 vb = vec4((P.xy + vec2(-u_radius,u_radius)) * u_scale, P.z * u_scale,1);
  gl_Position = u_pMatrix * vb;
  v_texCoord = vec2(0.0, 1.0);
  v_particleColor = u_color;
  v_particleColor.w = vert[0].transparency;
  EmitVertex();  

  //TOP RIGHT
  vec4 vc = vec4((P.xy + vec2(u_radius, u_radius)) * u_scale, P.z * u_scale,1);
  gl_Position = u_pMatrix *  vc;
  v_texCoord = vec2(1.0, 1.0);
  v_particleColor = u_color;
  v_particleColor.w = vert[0].transparency;
  EmitVertex();  
  
  EndPrimitive();  // Send primitives to fragment shader
}   