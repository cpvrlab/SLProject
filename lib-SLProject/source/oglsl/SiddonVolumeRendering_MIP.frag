//#############################################################################
//  File:      SiddonVolumeRendering_MIP.frag
//  Purpose:   GLSL fragment shader performing volume rendering using Siddon's
//             algorithm. The resulting color is calculated using maximum
//             intensity projection.
//  Author:    Manuel Frischknecht
//  Date:      March 2015
//  Copyright: 2002-2015 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#version 130

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision mediump float;
#endif

varying     vec3       v_raySource;      //The source coordinate of the view ray (model coordinates)

uniform     vec3       u_voxelScale;     //Voxel scaling coefficients
uniform     vec3       u_eyePosition;    //The position of the camera (model coordinates)

uniform     sampler3D  u_volume;         //The volume texture
uniform     sampler1D  u_TfLut;          //A LUT for the transform function

vec3 findRayDestination(vec3 raySource, vec3 rayDirection)
{
    // We are looking for the point on raySource + f*rayDirection with either x, y or z set to -1 or 1
    // Of all possibilities, we want the smallest positive (non-zero) factor f (i.e. the first intersection
    // with any wall of the cube).

    //Avoid infinity in the division below (in case the direction is parallel to a plane)
    vec3 d = rayDirection +  vec3(isinf(1.0f/rayDirection))*0.001f;

    // Coefficient-wise resolve source + f*direction = X (with X = +-1)
    // -> f = (X-source)/direction
    vec3 f1 = (vec3( 1.0f)-raySource)/d;
    vec3 f2 = (vec3(-1.0f)-raySource)/d;

    //Mask out negative values
    f1 *= step(0.0f,f1);
    f2 *= step(0.0f,f2);

    //Assure that no zeros are picked (4 > sqrt(3*(2^2)), which is the longest possible distance in the cube)
    f1 += vec3(isinf(1.0f/f1))*4;
    f2 += vec3(isinf(1.0f/f2))*4;

    //Find the smallest factor
    f1 = min(f1,f2);
    float f = min(min(f1.x, f1.y), f1.z);

    return raySource + f*rayDirection;
}

void main()
{
    vec3 source = v_raySource;
    vec3 direction = normalize(source - u_eyePosition);
    vec3 destination = findRayDestination(source,direction);

    //'scale' is used to scale the texture in case of non-uniform voxel size
    vec3 size = vec3(textureSize(u_volume, 0))*u_voxelScale;
    float maxLength = max(max(size.x, size.y), size.z);

    //'cuboidScaling' is used to scale the texture-space coordinates for non-cubic textures
    vec3 cuboidScaling = vec3(maxLength)/size;
    //
    source      = 0.5f*cuboidScaling*source+vec3(.5f);
    destination = 0.5f*cuboidScaling*destination+vec3(.5f);
    direction   = destination-source;
	
    float currentPosition = 0.0f;
    vec3 delta = abs(1.0f/(direction*size)); //Distance (in %) between two axis intersections along the ray (per axis)
    vec3 nextIntersections = fract(vec3(1.0f)+fract(source*size)); //Source position in the first voxel (0..1 for each axis)
    nextIntersections = step(0.0f,direction)-nextIntersections; //Distance to the next voxel boundary for each axis (in % of voxel width)
	nextIntersections += vec3(1.0f) - abs(sign(nextIntersections)); //If we're at 0.0 for any dimension, pick the whole voxel width (to avoid a first zero-step)
    nextIntersections /= size; //Distance to the next voxel boundary for each axis in % of volume length
    nextIntersections = nextIntersections/direction; //Distance to the next voxel boundary for each axis in % along the ray

    int i = 0;
    gl_FragColor = vec4(0.0f);
    while(currentPosition < 1.0f)
    {
        //if (i++ > 10000) break; //Debugging Security: Avoid driver crash in an endless loop

        // Basic Siddon's algorithm:
        // see: Accelerated ray tracing for radiotherapy dose calculations on a GPU
        // (http://dare.uva.nl/document/2/101986)

        float segmentPosition = 0.0f;
        float segmentLength = 0.0f;
        //Pick the next intersection (which is the smallest factor in nextIntersections)
        if (nextIntersections.x < nextIntersections.y && nextIntersections.x < nextIntersections.z)
        {
            //Calculate the center and length of the ray segment intersecting with the current voxel
            segmentPosition = (currentPosition+nextIntersections.x)/2;
            segmentLength   = nextIntersections.x - currentPosition;
            //Update the current position to the determined intersection
            currentPosition  = nextIntersections.x;
            //Update the position of the next intersection with the X plane
            nextIntersections.x += delta.x;
        }
        else if (nextIntersections.y  < nextIntersections.z)
        {
            segmentPosition = (currentPosition+nextIntersections.y)/2;
            segmentLength   = nextIntersections.y - currentPosition;
            currentPosition  = nextIntersections.y;
            nextIntersections.y += delta.y;
        }
        else
        {
            segmentPosition = (currentPosition+nextIntersections.z)/2;
            segmentLength   = nextIntersections.z - currentPosition;
            currentPosition  = nextIntersections.z;
            nextIntersections.z += delta.z;
        }
		
        //Normalize the fragment length (with 1.0f being the voxel width)
        segmentLength = length(segmentLength*size*direction);

        //Calculate the position (i.e. the middle) of the line fragment in the texture space
        vec3 position = source+currentPosition*direction;
        //The voxel can be read directly from there assuming we're using GL_NEAREST as interpolation method
        vec4 voxel = texture3D(u_volume, position);

        voxel = vec4(1.0f,1.0f,1.0f, voxel.r);
        //Adjust the voxel's alpha value by using the segment length ratio calculated earlier
        voxel.a   *= segmentLength;

        //Scale the color by it's alpha value
        voxel.rgb *= voxel.a;

        //Only pick the voxel's color if it's alpha value is greater as the one currently cached
        if (gl_FragColor.a < voxel.a)
            gl_FragColor = voxel;
    }

    gl_FragColor.rgb *= gl_FragColor.a;
    gl_FragColor.a = 1.0f;
}
