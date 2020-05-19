#pragma once

#include "math/SLMat4.h"

//-----------------------------------------------------------------------------
struct UniformBufferObject
{
    SLMat4f model;
    SLMat4f view;
    SLMat4f proj;
};
//-----------------------------------------------------------------------------
