#pragma once

#include "Device.h"

#include <array>
#include <vector>

class Swapchain;

class DescriptorPool
{
public:
    DescriptorPool(Device& device, Swapchain& swapchain);

public:
    Device&          device;
    VkDescriptorPool handle{VK_NULL_HANDLE};
};
