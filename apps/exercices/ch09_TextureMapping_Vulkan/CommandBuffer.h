#pragma once

#include "Framebuffer.h"
#include "RenderPass.h"
#include "IndexBuffer.h"
#include "Pipeline.h"
#include "DescriptorSet.h"
#include "VertexBuffer.h"

struct Vertex;
class Device;
class Swapchain;
class DescriptorSet;
class Buffer;
class Pipeline;

//-----------------------------------------------------------------------------
class CommandBuffer
{
public:
    CommandBuffer(Device& device) : _device{device} {};
    void destroy();

    VkResult begin();
    void     end();
    void     setVertices(Swapchain&     swapchain,
                         Framebuffer&   framebuffer,
                         RenderPass&    renderPass,
                         Buffer&        vertexBuffer,
                         Buffer&        indexBuffer,
                         Pipeline&      pipeline,
                         DescriptorSet& descriptorSet,
                         int            indicesSize);

    // Getter
    Device&                 device() const { return _device; }
    VkCommandBuffer         handle() const { return _handle; }
    vector<VkCommandBuffer> handles() const { return _handles; }

private:
    Device&                 _device;
    VkCommandBuffer         _handle{VK_NULL_HANDLE};
    vector<VkCommandBuffer> _handles;
};
//-----------------------------------------------------------------------------
