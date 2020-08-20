#include "CommandBuffer.h"

#include <array>

void CommandBuffer::destroy()
{
    if (_handle != VK_NULL_HANDLE)
        vkFreeCommandBuffers(_device.handle(), _device.commandPool(), 1, &_handle);

    for (size_t i = 0; i < _handles.size(); i++)
        if (_handles[i] != VK_NULL_HANDLE)
            vkFreeCommandBuffers(_device.handle(), _device.commandPool(), 1, &_handles[i]);
}

VkResult CommandBuffer::begin()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool        = _device.commandPool();
    allocInfo.commandBufferCount = 1;

    VkResult result = vkAllocateCommandBuffers(_device.handle(), &allocInfo, &_handle);
    ASSERT_VULKAN(result, "Failed to allocate commandBuffer!");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    return vkBeginCommandBuffer(_handle, &beginInfo);
}

void CommandBuffer::end()
{
    vkEndCommandBuffer(_handle);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &_handle;

    vkQueueSubmit(_device.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(_device.graphicsQueue());

    vkFreeCommandBuffers(_device.handle(), _device.commandPool(), 1, &_handle);
}

void CommandBuffer::setVertices(Swapchain& swapchain, Framebuffer& framebuffer, RenderPass& renderPass, Buffer& vertexBuffer, Buffer& indexBuffer, Pipeline& pipeline, DescriptorSet& descriptorSet, const int indicesSize)
{
    _handles.resize(framebuffer.handle().size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = _device.commandPool();
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)_handles.size();

    VkResult result = vkAllocateCommandBuffers(_device.handle(), &allocInfo, _handles.data());
    ASSERT_VULKAN(result, "Failed to allocate command buffers");

    for (size_t i = 0; i < _handles.size(); i++)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        result = vkBeginCommandBuffer(_handles[i], &beginInfo);
        ASSERT_VULKAN(result, "Failed to begin recording command buffer");

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass        = renderPass.handle();
        renderPassInfo.framebuffer       = framebuffer.handle()[i];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapchain.extent();

        array<VkClearValue, 2> clearValues{};
        clearValues[0].color        = {0.0f, 0.0f, 0.0f, 1.0f};
        clearValues[1].depthStencil = {1.0f, 0};

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues    = clearValues.data();

        vkCmdBeginRenderPass(_handles[i],
                             &renderPassInfo,
                             VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(_handles[i],
                          VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipeline.graphicsPipeline());
        // TODO: Fill array of vertexBuffers + indexBuffers
        VkBuffer     vertexBuffers[] = {vertexBuffer.handle()};
        VkDeviceSize offsets[]       = {0};
        vkCmdBindVertexBuffers(_handles[i],
                               0,
                               1,
                               vertexBuffers,
                               offsets);
        vkCmdBindIndexBuffer(_handles[i],
                             indexBuffer.handle(),
                             0,
                             VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(_handles[i],
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline.pipelineLayout(),
                                0,
                                1,
                                &descriptorSet.handles()[i],
                                0,
                                nullptr);
        vkCmdDrawIndexed(_handles[i],
                         static_cast<uint32_t>(indicesSize),
                         1,
                         0,
                         0,
                         0);
        vkCmdEndRenderPass(_handles[i]);

        result = vkEndCommandBuffer(_handles[i]);
        ASSERT_VULKAN(result, "Failed to record command buffer");
    }
}

void CommandBuffer::setVertices(Swapchain& swapchain, Framebuffer& framebuffer, RenderPass& renderPass, vector<Buffer*> vertexBuffer, vector<Buffer*> indexBuffer, vector<Pipeline*> pipeline, vector<DescriptorSet*> descriptorSet, vector<int> indicesSize, RangeManager& rangeManager)
{
    _handles.resize(framebuffer.handle().size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = _device.commandPool();
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)_handles.size();

    VkResult result = vkAllocateCommandBuffers(_device.handle(), &allocInfo, _handles.data());
    ASSERT_VULKAN(result, "Failed to allocate command buffers");

    for (size_t i = 0; i < _handles.size(); i++)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        result = vkBeginCommandBuffer(_handles[i], &beginInfo);
        ASSERT_VULKAN(result, "Failed to begin recording command buffer");

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass        = renderPass.handle();
        renderPassInfo.framebuffer       = framebuffer.handle()[i];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapchain.extent();

        array<VkClearValue, 2> clearValues{};
        clearValues[0].color        = {0.0f, 0.0f, 0.0f, 1.0f};
        clearValues[1].depthStencil = {1.0f, 0};

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues    = clearValues.data();

        vkCmdBeginRenderPass(_handles[i],
                             &renderPassInfo,
                             VK_SUBPASS_CONTENTS_INLINE);

        for (int j = 0; j < pipeline.size(); j++)
        {
            vkCmdBindPipeline(_handles[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipeline[j]->graphicsPipeline());

            // TODO: Fill array of vertexBuffers + indexBuffers
            // VkBuffer* vertexBuffers = new VkBuffer[rangeManager.getDiff(j)];
            // for (int k = rangeManager.getMin(j); k < rangeManager.getMax(j); k++)
            //     vertexBuffers[k] = vertexBuffer[k]->handle();
            vector<VkBuffer> vertexBuffers = vector<VkBuffer>(rangeManager.getDiff(j));
            int              zeroCounter   = 0;
            for (int k = rangeManager.getMin(j); k < rangeManager.getMax(j); k++)
            {
                vertexBuffers[zeroCounter] = vertexBuffer[k]->handle();
                zeroCounter++;
            }
            // VkBuffer     vertexBuffers[] = {vertexBuffer[j]->handle()};
            VkDeviceSize offsets[1] = {0};
            vkCmdBindVertexBuffers(_handles[i],
                                   0,
                                   1,
                                   vertexBuffers.data(),
                                   offsets);

            // TODO: Create List for which pipeline owns which vertexBuffer
            for (int k = rangeManager.getMin(j); k < rangeManager.getMax(j); k++)
            {
                vkCmdBindIndexBuffer(_handles[i],
                                     indexBuffer[k]->handle(),
                                     0,
                                     VK_INDEX_TYPE_UINT32);
                vkCmdBindDescriptorSets(_handles[i],
                                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipeline[j]->pipelineLayout(),
                                        0,
                                        1,
                                        &descriptorSet[k]->handles()[i],
                                        0,
                                        nullptr);
                vkCmdDrawIndexed(_handles[i],
                                 static_cast<uint32_t>(indicesSize[k]),
                                 1,
                                 0,
                                 0,
                                 0);
            }

            // delete[] vertexBuffers;
            // vertexBuffers = nullptr;
        }

        vkCmdEndRenderPass(_handles[i]);

        result = vkEndCommandBuffer(_handles[i]);
        ASSERT_VULKAN(result, "Failed to record command buffer");
    }
}
