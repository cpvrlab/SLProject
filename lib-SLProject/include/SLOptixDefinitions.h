//
// Created by nic on 24.10.19.
//

#ifndef SLPROJECT_SLOPTIXDEFINITIONS_H
#define SLPROJECT_SLOPTIXDEFINITIONS_H

#include <vector_types.h>

struct Params
{
    uchar4* image;
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

#endif //SLPROJECT_SLOPTIXDEFINITIONS_H
