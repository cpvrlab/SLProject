//
// Created by nic on 24.10.19.
//

#ifndef SLPROJECT_SLOPTIXDEFINITIONS_H
#define SLPROJECT_SLOPTIXDEFINITIONS_H

#include <vector_types.h>

struct Params
{
    uchar3* image;
    unsigned int image_width;
};

struct CameraData
{
    float r,g,b;
};

struct Light
{
    float3 position;
    float3 color;
};

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<CameraData>   RayGenSbtRecord;
typedef SbtRecord<int>   MissSbtRecord;
typedef SbtRecord<int>   HitSbtRecord;

#endif //SLPROJECT_SLOPTIXDEFINITIONS_H
