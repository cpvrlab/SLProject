//
// Created by nic on 24.10.19.
//

#ifndef SLPROJECT_SLOPTIXDEFINITIONS_H
#define SLPROJECT_SLOPTIXDEFINITIONS_H

#include <vector_types.h>
#include <optix_types.h>

struct Material
{
    float3 diffuse_color;
    float3 ambient_color;
    float3 specular_color;
    float shininess;
    float kr;
    float kt;
    float kn;
};

struct Light
{
    float3 position;
    float3 color;
};

struct Params
{
    uchar4*                 image;
    unsigned int            width;
    unsigned int            height;

    int                     max_depth;
    float                   scene_epsilon;

    OptixTraversableHandle  handle;

    Light*                  lights;
    int                     numLights;
};

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
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

struct MissData
{
    float3 bg_color;
};

struct HitData
{
    Material material;

    int sbtIndex;
    float3* normals;
    int3* indices;
};

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<CameraData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitData>      HitSbtRecord;

#endif //SLPROJECT_SLOPTIXDEFINITIONS_H
