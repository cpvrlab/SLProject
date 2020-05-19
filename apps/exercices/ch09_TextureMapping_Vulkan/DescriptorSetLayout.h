#pragma once

#include "Device.h"

//-----------------------------------------------------------------------------
class DescriptorSetLayout
{
public:
    DescriptorSetLayout(Device& device);
    void destroy();

public:
    Device&               device;
    VkDescriptorSetLayout handle{VK_NULL_HANDLE};
};
//-----------------------------------------------------------------------------
