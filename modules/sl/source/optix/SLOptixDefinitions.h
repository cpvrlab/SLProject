//#############################################################################
//  File:      SLOptixDefinitions.h
//  Authors:   Nic Dorner
//  Date:      October 2019
//  Authors:   Nic Dorner
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_HAS_OPTIX
#    ifndef SLOPTIXDEFINITIONS_H
#        define SLOPTIXDEFINITIONS_H
#        include <vector_types.h>
#        include <optix_types.h>
#        include <cuda.h>
#        include <curand_kernel.h>

//------------------------------------------------------------------------------
//! Optix ray tracing sample struct
struct ortSamples
{
    unsigned int samplesX;
    unsigned int samplesY;
};
//------------------------------------------------------------------------------
//! Optix ray tracing material information struct
struct ortMaterial
{
    float4 diffuse_color;
    float4 ambient_color;
    float4 specular_color;
    float4 transmissiv_color;
    float4 emissive_color;
    float  shininess;
    float  kr;
    float  kt;
    float  kn;
};
//------------------------------------------------------------------------------
//! Optix ray tracing light information struct
struct ortLight
{
    float4     diffuse_color;
    float4     ambient_color;
    float4     specular_color;
    float3     position;
    float      spotCutOffDEG;
    float      spotExponent;
    float      spotCosCut;
    float3     spotDirWS;
    float      kc;
    float      kl;
    float      kq;
    ortSamples samples;
    float      radius;
};
//------------------------------------------------------------------------------
struct ortParams
{
    float4*      image;
    unsigned int width;
    unsigned int height;

    int   max_depth;
    float scene_epsilon;

    OptixTraversableHandle handle;

    union
    {
        struct
        {
            ortLight*    lights;
            unsigned int numLights;
            float4       globalAmbientColor;
        };

        struct
        {
            unsigned int samples;
            unsigned int seed;
            curandState* states;
        };
    };
};
//------------------------------------------------------------------------------
//! Optix ray tracing ray type enumeration
enum ortRayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};
//------------------------------------------------------------------------------
//! Optix ray tracing camera info for pinhole camera
struct ortCamera
{
    float3 eye;
    float3 U;
    float3 V;
    float3 W;
};
//------------------------------------------------------------------------------
//! Optix ray tracing camera info for lens camera
struct ortLensCamera
{
    ortCamera  camera;
    ortSamples samples;
    float      lensDiameter;
};
//------------------------------------------------------------------------------
//! Optix ray tracing intersection miss data
struct ortMissData
{
    float4 bg_color;
};
//------------------------------------------------------------------------------
//! Optix ray tracing intersection hit data
struct ortHitData
{
    ortMaterial material;
    CUtexObject textureObject;
    int         sbtIndex;
    float3*     normals;
    short3*     indices;
    float2*     texCords;
};
//------------------------------------------------------------------------------
template<typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
//------------------------------------------------------------------------------
typedef SbtRecord<ortCamera>     RayGenClassicSbtRecord;
typedef SbtRecord<ortLensCamera> RayGenDistributedSbtRecord;
typedef SbtRecord<ortMissData>   MissSbtRecord;
typedef SbtRecord<ortHitData>    HitSbtRecord;
//------------------------------------------------------------------------------
#    endif // SLOPTIXDEFINITIONS_H
#endif     // SL_HAS_OPTIX
