#include "CommandBuffer.h"

VkResult CommandBuffer::begin()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool        = device.commandPool;
    allocInfo.commandBufferCount = 1;

    VkResult result = vkAllocateCommandBuffers(device.handle, &allocInfo, &handle);
    ASSERT_VULKAN(result, "Failed to allocate commandBuffer!");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    return vkBeginCommandBuffer(handle, &beginInfo);
}

void CommandBuffer::end()
{
    vkEndCommandBuffer(handle);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &handle;

    vkQueueSubmit(device.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(device.graphicsQueue);

    vkFreeCommandBuffers(device.handle, device.commandPool, 1, &handle);
}

void CommandBuffer::setVertices(const std::vector<Vertex>& vertices, Swapchain& swapchain, Framebuffer& framebuffer, RenderPass& renderPass, Buffer& vertexBuffer, Buffer& indexBuffer, Pipeline& pipeline, DescriptorSet& descriptorSet, int indicesSize)
{
    handles.resize(framebuffer.handle.size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = device.commandPool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)handles.size();

    VkResult result = vkAllocateCommandBuffers(device.handle, &allocInfo, handles.data());
    ASSERT_VULKAN(result, "Failed to allocate command buffers");

    for (size_t i = 0; i < handles.size(); i++)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        result = vkBeginCommandBuffer(handles[i], &beginInfo);
        ASSERT_VULKAN(result, "Failed to begin recording command buffer");

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass        = renderPass.handle;
        renderPassInfo.framebuffer       = framebuffer.handle[i];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapchain.extent;

        VkClearValue clearColor        = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues    = &clearColor;

        vkCmdBeginRenderPass(handles[i],
                             &renderPassInfo,
                             VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(handles[i],
                          VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipeline.graphicsPipeline);
        // VkBuffer     vertexBuffer    = vertexBuffer.handle;
        VkBuffer     vertexBuffers[] = {vertexBuffer.handle};
        VkDeviceSize offsets[]       = {0};
        vkCmdBindVertexBuffers(handles[i],
                               0,
                               1,
                               vertexBuffers,
                               offsets);
        vkCmdBindIndexBuffer(handles[i],
                             indexBuffer.handle,
                             0,
                             VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(handles[i],
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline.pipelineLayout,
                                0,
                                1,
                                &descriptorSet.handles[i],
                                0,
                                nullptr);
        vkCmdDrawIndexed(handles[i],
                         static_cast<uint32_t>(indicesSize),
                         1,
                         0,
                         0,
                         0);
        vkCmdEndRenderPass(handles[i]);

        result = vkEndCommandBuffer(handles[i]);
        ASSERT_VULKAN(result, "Failed to record command buffer");
    }
}
