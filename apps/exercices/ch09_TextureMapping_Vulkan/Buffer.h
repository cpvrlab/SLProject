#pragma once

#include <Device.h>
#include <CommandBuffer.h>

struct Vertex;

//-----------------------------------------------------------------------------
class Buffer
{
public:
    Buffer(Device& device) : device{device} {};
    void     destroy();
    void     free();
    uint32_t findMemoryType(uint32_t              typeFilter,
                            VkMemoryPropertyFlags properties);
    void     copy(Buffer src, VkDeviceSize size);
    void     createVertexBuffer(const vector<Vertex>& vertices);
    void     createIndexBuffer(const vector<uint16_t> indices);

public:
    void createBuffer(VkDeviceSize,
                      VkBufferUsageFlags,
                      VkMemoryPropertyFlags);

public:
    Device&        device;
    VkBuffer       handle{VK_NULL_HANDLE};
    VkDeviceMemory memory;
};
//-----------------------------------------------------------------------------
