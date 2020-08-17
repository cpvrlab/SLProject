#pragma once

#include "Device.h"
#include <array>
#include <vector>

class Swapchain;

//-----------------------------------------------------------------------------
class DescriptorPool
{
public:
    DescriptorPool(Device& device, Swapchain& swapchain);
    void destroy();

    // Getter
    VkDescriptorPool handle() const { return _handle; }

private:
    Device&          _device;
    VkDescriptorPool _handle{VK_NULL_HANDLE};
};
//-----------------------------------------------------------------------------
