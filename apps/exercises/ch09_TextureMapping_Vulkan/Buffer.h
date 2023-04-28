#pragma once

#include <Device.h>
#include <CommandBuffer.h>
#include <SLVec4.h>

struct Vertex;

//-----------------------------------------------------------------------------
class Buffer
{
public:
    Buffer(Device& device) : _device{device} {};
    void            destroy();
    static uint32_t findMemoryType(Device&               device,
                                   uint32_t              typeFilter,
                                   VkMemoryPropertyFlags properties);
    void            copy(Buffer src, VkDeviceSize size);
    void            createVertexBuffer(const vector<Vertex>& vertices);
    void            createVertexBuffer(const SLVVec3f pos,
                                       const SLVVec3f norm,
                                       const SLVVec2f texCoord,
                                       const SLVCol4f color,
                                       const size_t   size);
    void            createIndexBuffer(const SLVuint indices);
    void            createBuffer(VkDeviceSize, VkBufferUsageFlags, VkMemoryPropertyFlags);

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
