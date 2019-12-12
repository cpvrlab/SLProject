//#############################################################################
//  File:      VolumeRenderingRayCast.frag
//  Purpose:   GLSL fragment shader performing volume rendering using a simple
//             sampling method along the view ray. The resulting color is
//             calculated using a transfer function (via a lookup table).
//  Author:    Manuel Frischknecht
//  Date:      March 2015
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
precision highp sampler2D;
precision highp sampler3D;
#endif

varying vec3       v_raySource;     // The source coordinate of the view ray (model coordinates)

uniform mat4       u_invMvMatrix;   // inverse modelView matrix = view matrix
uniform float      u_volumeX;       // 3D texture width
uniform float      u_volumeY;       // 3D texture height
uniform float      u_volumeZ;       // 3D texture depth
uniform float      u_oneOverGamma;  // 1.0f / Gamma correction value
uniform sampler3D  u_texture0;      // The 3D volume texture
uniform sampler2D  u_texture1;      // The 1D LUT for the transform function

vec3 findRayDestination(vec3 raySource, vec3 rayDirection)
{
    // We are looking for the point on raySource + f*rayDirection with either x, y or z set to -1 or 1
    // Of all possibilities, we want the smallest positive (non-zero) factor f (i.e. the first intersection
    // with any wall of the cube).

    //Avoid infinity in the division below (in case the direction is parallel to a plane)
    vec3 d = rayDirection; // +  vec3(isinf(1.0/rayDirection))*0.001;

    // Coefficient-wise resolve source + f*direction = X (with X = +-1)
    // -> f = (X-source)/direction
    vec3 f1 = (vec3( 1.0)-raySource)/d;
    vec3 f2 = (vec3(-1.0)-raySource)/d;

    //Mask out negative values
    f1 *= step(0.0,f1);
    f2 *= step(0.0,f2);
	
    //Assure that no zeros are picked (4 > sqrt(3*(2^2)), which is the longest possible distance in the cube)
    //f1 += vec3(isinf(1.0/f1))*4;
    f1 += (vec3(1.0)-vec3(step(0.0001,f1)))*4.0;
    //f2 += vec3(isinf(1.0/f2))*4;
    f2 += (vec3(1.0)-vec3(step(0.0001,f2)))*4.0;

    //Find the smallest factor
    f1 = min(f1,f2);
    float f = min(min(f1.x, f1.y), f1.z);

    return raySource + f*rayDirection;
}

void main()
{
    vec3 source = v_raySource;
    vec4 eye = u_invMvMatrix * vec4(0.0, 0.0, 0.0, 1.0);
    vec3 direction = normalize(source - eye.xyz);
    vec3 destination = findRayDestination(source,direction);

    vec3 size = vec3(u_volumeX, u_volumeY, u_volumeZ);
    float maxLength = max(max(size.x, size.y), size.z);

    //'cuboidScaling' is used to scale the texture-space coordinates for non-cubic textures
    vec3 cuboidScaling = float(maxLength)/vec3(size);

    //Calculate the distance between samples
    float step_dist = 1.0/(sqrt(3.0)*float(maxLength));

    //Project the source and destination coordinates from cube space into texture space
    source      = 0.5*cuboidScaling*source+vec3(.5);
    destination = 0.5*cuboidScaling*destination+vec3(.5);
    direction   = destination-source;

    //Set direction to the length of a single step
    float distance = length(direction);
    direction = normalize(direction)*step_dist;
	
    //Calculate the amount of steps to loop through
    int num_steps = int(floor(distance/(step_dist)));
    vec3 position = source;
    gl_FragColor = vec4(0.0);

    for (int i = 0; i < num_steps; ++i) //Step along the view ray
    {
        //The voxel can be read directly from there assuming we're using GL_NEAREST as interpolation method
        vec4 voxel = texture3D(u_texture0, position);

        //Transform the read pixel with the 1D transform function lookup table
        voxel = texture2D(u_texture1, vec2(voxel.r, 0.0));

        //Scale the color addend by it's alpha value
        voxel.rgb *= voxel.a;

        gl_FragColor = (1.0-gl_FragColor.a)*voxel + gl_FragColor;

        //Jump out of the loop if the cumulated alpha is above a threshold
        if (gl_FragColor.a > 0.99)
            break;

        //Set the position to the next step
        position += direction;
    }
    gl_FragColor.a = 1.0;

    // Apply gamma correction
    gl_FragColor.rgb = pow(gl_FragColor.rgb, vec3(u_oneOverGamma));
}
