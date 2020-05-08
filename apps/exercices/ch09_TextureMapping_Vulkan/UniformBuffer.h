#pragma once

#include "Device.h"
#include "Buffer.h"
#include "UniformBufferObject.h"
#include "Swapchain.h"

class UniformBuffer
{
public:
    UniformBuffer(Device& device, Swapchain& swapchain);
    void update(uint32_t currentImage);

public:
    Device&              device;
    Swapchain&           swapchain;
    std::vector<Buffer*> buffers;
};
