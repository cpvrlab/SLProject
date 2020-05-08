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
