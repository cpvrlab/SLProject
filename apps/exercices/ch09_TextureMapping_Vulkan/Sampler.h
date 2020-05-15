#pragma once

#include "Device.h"

class Sampler
{
public:
    Sampler(Device& device);
    void destroy();

public:
    Device&   device;
    VkSampler handle{VK_NULL_HANDLE};
};
