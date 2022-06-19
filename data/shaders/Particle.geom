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
layout (triangle_strip, max_vertices = 4) out;    // Primitives that we will output and number of vertex that will be output

in vertex {
    float transparency; // Transparency of a particle
    float rotation;     // Rotation of a particle
    float size;
} vert[];

uniform mat4 u_pMatrix;     // Projection matrix

uniform vec4 u_color;       // Particle color
uniform float u_scale;      // Particle scale
uniform float u_radius;     // Particle radius

out vec4 v_particleColor;   // The resulting color per vertex
out vec2 v_texCoord;        // Texture coordinate at vertex

// Plot a line on Y using a value between 0.0-1.0
float plot(float age, float ttl) {    
    return smoothstep(0.0, 1.0, age/ttl);
}

//-----------------------------------------------------------------------------
void main (void)
{
  float scale = u_scale;
  scale *= vert[0].size;
  float radius = u_radius * scale;

  vec4 P = gl_in[0].gl_Position;    // Position of the point that we received
  mat2 rot = mat2(cos(vert[0].rotation),-sin(vert[0].rotation),sin(vert[0].rotation),cos(vert[0].rotation)); // Matrix of rotation
  // Create 4 points to create two triangle to draw one particle

  vec4 color = u_color;              // Particle color
  color.w *= vert[0].transparency;   // Apply transparency

  //BOTTOM LEFT
  vec4 va = vec4(P.xy + (rot * vec2(-radius, -radius)), P.z, 1); //Position in view space
  gl_Position = u_pMatrix * va; // Calculate position in clip space
  v_texCoord = vec2(0.0, 0.0);  // Texture coordinate
  v_particleColor = color;
  EmitVertex();  
  
  //BOTTOM RIGHT
  vec4 vd = vec4(P.xy + (rot * vec2(radius, -radius)), P.z,1);
  gl_Position = u_pMatrix * vd;
  v_texCoord = vec2(1.0, 0.0);
  v_particleColor = color;
  EmitVertex();  

  //TOP LEFT
  vec4 vb = vec4(P.xy + (rot * vec2(-radius,radius)) , P.z,1);
  gl_Position = u_pMatrix * vb;
  v_texCoord = vec2(0.0, 1.0);
  v_particleColor = color;
  EmitVertex();  

  //TOP RIGHT
  vec4 vc = vec4(P.xy + (rot *vec2(radius, radius)), P.z,1);
  gl_Position = u_pMatrix *  vc;
  v_texCoord = vec2(1.0, 1.0);
  v_particleColor = color;
  EmitVertex();  
  
  EndPrimitive();  // Send primitives to fragment shader
}   