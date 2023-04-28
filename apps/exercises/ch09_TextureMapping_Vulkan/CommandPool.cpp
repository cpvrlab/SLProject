#include "CommandPool.h"
/*
CommandPool::CommandPool(Device& device) : device{device}
{
    createCommandPool();
}

// void CommandPool::addBuffer(CommandBuffer* commandBuffer)
// {
//     commandBuffers.push_back(commandBuffer);
// }

void CommandPool::createCommandPool()
{
    QueueFamilyIndices queueFamilyIndices = device.findQueueFamilies(device.physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;

    VkResult result = vkCreateCommandPool(device.handle, &poolInfo, nullptr, &handle);
    ASSERT_VULKAN(result, "Failed to create command pool");
}
*/
