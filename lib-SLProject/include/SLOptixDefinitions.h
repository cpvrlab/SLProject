//
// Created by nic on 24.10.19.
//

#ifndef SLPROJECT_SLOPTIXDEFINITIONS_H
#define SLPROJECT_SLOPTIXDEFINITIONS_H

#include <vector_types.h>
#include <optix_types.h>
#include <cuda.h>
#include <curand_kernel.h>

struct Line
{
    float3 p1;
    float3 p2;
};

struct Ray
{
    Line line;
    float4 color;
};

struct Samples
{
    unsigned int samplesX;
    unsigned int samplesY;
};

struct Material
{
    float4  diffuse_color;
    float4  ambient_color;
    float4  specular_color;
    float4  transmissiv_color;
    float4  emissive_color;
    float   shininess;
    float   kr;
    float   kt;
    float   kn;
};

struct Light
{
    float4  diffuse_color;
    float4  ambient_color;
    float4  specular_color;
    float3  position;
    float   spotCutOffDEG;
    float   spotExponent;
    float   spotCosCut;
    float3  spotDirWS;
    float   kc;
    float   kl;
    float   kq;
    Samples samples;
    float   radius;
};

struct Params
{
    float4*                 image;
    unsigned int            width;
    unsigned int            height;

    int                     max_depth;
    float                   scene_epsilon;

    OptixTraversableHandle  handle;

    union {
        struct {
            Light*                  lights;
            unsigned int            numLights;
            float4                  globalAmbientColor;
        };

        struct {
            unsigned int samples;
            unsigned int seed;
            curandState*    states;
        };
    };

    Ray*   rays;
};

enum RayType
{
    RAY_TYPE_RADIANCE   = 0,
    RAY_TYPE_OCCLUSION  = 1,
    RAY_TYPE_COUNT
};

struct CameraData
{
    float3       eye;
    float3       U;
    float3       V;
    float3       W;
};

struct LensCameraData
{
    CameraData camera;
    Samples samples;
    float lensDiameter;
};

struct MissData
{
    float4 bg_color;
};

struct HitData
{
    union
    {
        Line    line;
    } geometry;

    Material    material;
    CUtexObject textureObject;

    int         sbtIndex;
    float3*     normals;
    short3*     indices;
    float2*     texCords;
};

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<CameraData>       RayGenClassicSbtRecord;
typedef SbtRecord<LensCameraData>   RayGenDistributedSbtRecord;
typedef SbtRecord<MissData>         MissSbtRecord;
typedef SbtRecord<HitData>          HitSbtRecord;

#endif //SLPROJECT_SLOPTIXDEFINITIONS_H
