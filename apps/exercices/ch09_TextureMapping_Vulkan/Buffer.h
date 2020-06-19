#pragma once

#include <Device.h>
#include <CommandBuffer.h>

struct Vertex;

//-----------------------------------------------------------------------------
class Buffer
{
public:
    Buffer(Device& device) : _device{device} {};
    void     destroy();
    void     free();
    uint32_t findMemoryType(uint32_t              typeFilter,
                            VkMemoryPropertyFlags properties);
    void     copy(Buffer src, VkDeviceSize size);
    void     createVertexBuffer(const vector<Vertex>& vertices);
    void     createIndexBuffer(const vector<uint16_t> indices);
    void     createBuffer(VkDeviceSize, VkBufferUsageFlags, VkMemoryPropertyFlags);

    // Getter
    Device&        device() const { return _device; }
    VkBuffer       handle() const { return _handle; }
    VkDeviceMemory memory() const { return _memory; }

private:
    Device&        _device;
    VkBuffer       _handle{VK_NULL_HANDLE};
    VkDeviceMemory _memory;
};
//-----------------------------------------------------------------------------
