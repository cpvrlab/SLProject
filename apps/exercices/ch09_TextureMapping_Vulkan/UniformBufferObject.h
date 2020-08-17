#ifndef UNIFORMBUFFEROBJECT_H
#define UNIFORMBUFFEROBJECT_H

#include "math/SLMat4.h"

//-----------------------------------------------------------------------------
struct UniformBufferObject
{
    SLMat4f model;
    SLMat4f view;
    SLMat4f proj;
};
//-----------------------------------------------------------------------------
#endif
