//#############################################################################
//  File:      VolumeRenderingSlicing.frag
//  Purpose:   Vertex shader for slice-based volume rendering. The slices are
//             created as multiple quads aligned in a cubic shape.
//             The samples (as computed by this shader) have to be combined
//             using an appropriate OpenGL blending function in order to 
//             correctly display a volume.
//  Author:    Manuel Frischknecht
//  Date:      March 2015
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision mediump float;
#endif

varying     vec4         v_texCoord;              //The sample position in texture coordinates

uniform     sampler3D    u_volume;                //The volume texture
uniform     sampler1D    u_TfLut;                 //A LUT for the transform function

void main()
{
	//Determine the density at the sample position of this fragment
    float density = texture3D(u_volume, v_texCoord.xyz).r;
	//Look up the according color in the LUT
    vec4 color = texture1D(u_TfLut, density);

    //Set the alpha value for pixels outside of the cube to zero
    color.a *= step(0.0,v_texCoord.x)*(1.0-step(1.0,v_texCoord.x))
             * step(0.0,v_texCoord.y)*(1.0-step(1.0,v_texCoord.y))
             * step(0.0,v_texCoord.z)*(1.0-step(1.0,v_texCoord.z));

    gl_FragColor = color;
}
