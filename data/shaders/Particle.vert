//#############################################################################
//  File:      ColorAttribute.vert
//  Purpose:   GLSL vertex program for simple per vertex attribute color
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;

//-----------------------------------------------------------------------------
in          vec4     a_position;       // Vertex position attribute
in          vec2     a_texCoord;       // Vertex texture coord. attribute


uniform mat4     u_mvpMatrix;       // modelview-projection matrix = projection * modelView
uniform vec4 color;
uniform vec3 offset; 
uniform vec3 cr_wSpace; 
uniform vec3 cu_wSpace; 
uniform float scale; 

out Vertex
{
    vec4     vv_particleColor;           // The resulting color per vertex
    vec2     vv_texCoord;        // texture coordinate at vertex
} vertex;
//-----------------------------------------------------------------------------
void main(void)
{   
    vertex.vv_particleColor = color;                        // pass color for interpolation

    // Set the texture coord. output for interpolated tex. coords.
    vertex.vv_texCoord = a_texCoord;

    //vec3 vP_wSpace = offset + cr_wSpace * a_position.x * scale + cu_wSpace * a_position.y * scale;

    //gl_Position = u_mvpMatrix * vec4(vP_wSpace, 1.0);
    gl_Position = u_mvpMatrix * vec4((a_position.xy * scale) + offset.xy,a_position.z + offset.z, 1.0);   // transform vertex position
    //gl_Position = u_mvpMatrix * vec4((a_position.xyz * scale) + offset, 1.0);   // transform vertex position

   // Set the transformes vertex position           
   //gl_Position = u_mvpMatrix * a_position;
}
//-----------------------------------------------------------------------------
