#pragma once

#include "Device.h"

//-----------------------------------------------------------------------------
class DescriptorSetLayout
{
public:
    DescriptorSetLayout(Device& device);
    void destroy();

    // Getter
    VkDescriptorSetLayout handle() { return _handle; }

private:
    Device&               _device;
    VkDescriptorSetLayout _handle{VK_NULL_HANDLE};
};
//-----------------------------------------------------------------------------
