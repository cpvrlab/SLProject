#pragma once

#include "Device.h"
#include "Swapchain.h"

#include <array>
#include <vector>

class DescriptorPool
{
public:
    DescriptorPool(Device& device, Swapchain& swapchain);

public:
    Device&          device;
    VkDescriptorPool handle{VK_NULL_HANDLE};
};
