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
    CommandBuffer(Device& device) : device{device} {};
    void destroy();

    VkResult begin();
    void     end();
    void     setVertices(const vector<Vertex>& vertices,
                         Swapchain&            swapchain,
                         Framebuffer&          framebuffer,
                         RenderPass&           renderPass,
                         Buffer&               vertexBuffer,
                         Buffer&               indexBuffer,
                         Pipeline&             pipeline,
                         DescriptorSet&        descriptorSet,
                         int                   indicesSize);

public:
    Device&                 device;
    VkCommandBuffer         handle{VK_NULL_HANDLE};
    vector<VkCommandBuffer> handles;
};
//-----------------------------------------------------------------------------
