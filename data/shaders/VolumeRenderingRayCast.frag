//#############################################################################
//  File:      VolumeRenderingRayCast.frag
//  Purpose:   GLSL fragment shader performing volume rendering using a simple
//             sampling method along the view ray. The resulting color is
//             calculated using a transfer function (via a lookup table).
//  Date:      March 2015
//  Authors:   Manuel Frischknecht, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

precision highp float;
precision highp sampler2D;
precision highp sampler3D;

//-----------------------------------------------------------------------------
in      vec3        v_raySource;     // The source coordinate of the view ray (model coordinates)

uniform mat4        u_mMatrix;             // Model matrix (object to world transform)
uniform mat4        u_vMatrix;             // View matrix (world to camera transform)
uniform float       u_volumeX;             // 3D texture width
uniform float       u_volumeY;             // 3D texture height
uniform float       u_volumeZ;             // 3D texture depth
uniform float       u_oneOverGamma;        // 1.0f / Gamma correction value
uniform sampler3D   u_matTextureDiffuse0;  // The 3D volume texture
uniform sampler2D   u_matTextureDiffuse1;  // The 1D LUT for the transform function

out     vec4        o_fragColor;      // output fragment color
//-----------------------------------------------------------------------------
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
    vec3 source      = v_raySource;
    mat4 invMvMatrix = inverse(u_vMatrix * u_mMatrix);
    vec4 eye         = invMvMatrix * vec4(0.0, 0.0, 0.0, 1.0);
    vec3 direction   = normalize(source - eye.xyz);
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
    o_fragColor = vec4(0.0);

    for (int i = 0; i < num_steps; ++i) //Step along the view ray
    {
        //The voxel can be read directly from there assuming we're using GL_NEAREST as interpolation method
        vec4 voxel = texture(u_matTextureDiffuse0, position);

        //If the texture coordinates are outside the range 0.0 - 1.0, set voxel to 0
        //This emulates GL_CLAMP_TO_BORDER, which is only available in version >= 3.2
        vec3 is_inside_lower = max(vec3(0.0), sign(position));
        vec3 is_inside_upper = max(vec3(0.0), sign(vec3(1.0) - position));
        vec3 is_inside_axes = is_inside_lower * is_inside_upper;
        voxel *= is_inside_axes.x * is_inside_axes.y * is_inside_axes.z;

        //Transform the read pixel with the 1D transform function lookup table
        voxel = texture(u_matTextureDiffuse1, vec2(voxel.r, 0.0));

        //Scale the color addend by it's alpha value
        voxel.rgb *= voxel.a;

        o_fragColor = (1.0-o_fragColor.a)*voxel + o_fragColor;

        //Jump out of the loop if the cumulated alpha is above a threshold
        if (o_fragColor.a > 0.99)
            break;

        //Set the position to the next step
        position += direction;
    }
    o_fragColor.a = 1.0;

    // Apply gamma correction
    o_fragColor.rgb = pow(o_fragColor.rgb, vec3(u_oneOverGamma));
}
