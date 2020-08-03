#ifndef COMMANDBUFFER_H
#define COMMANDBUFFER_H

#include "Framebuffer.h"
#include "RenderPass.h"
#include "IndexBuffer.h"
#include "Pipeline.h"
#include "DescriptorSet.h"
#include "VertexBuffer.h"
#include "RangeManager.h"

#include <array>

struct Vertex;
class Device;
class Swapchain;
class DescriptorSet;
class Buffer;
class Pipeline;
class Framebuffer;

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
                         const int      indicesSize);
    void     setVertices(Swapchain&             swapchain,
                         Framebuffer&           framebuffer,
                         RenderPass&            renderPass,
                         vector<Buffer*>        vertexBuffer,
                         vector<Buffer*>        indexBuffer,
                         vector<Pipeline*>      pipeline,
                         vector<DescriptorSet*> descriptorSet,
                         vector<int>            indicesSize,
                         RangeManager&          rangeManager);

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
#endif
