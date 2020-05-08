#pragma once

#include "Device.h"
#include "CommandBuffer.h"

class Buffer
{
public:
    Buffer(Device& device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags);
    void     free();
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void     copy(Buffer src, VkDeviceSize size);

private:
    void createBuffer(VkDeviceSize, VkBufferUsageFlags, VkMemoryPropertyFlags);

public:
    Device&        device;
    VkBuffer       handle{VK_NULL_HANDLE};
    VkDeviceMemory memory;
};
