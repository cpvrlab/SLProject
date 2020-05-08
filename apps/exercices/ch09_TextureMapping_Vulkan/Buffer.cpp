#include "Buffer.h"

Buffer::Buffer(Device& device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) : device{device}
{
    createBuffer(size, usage, properties);
}

void Buffer::free()
{
    vkDestroyBuffer(device.handle, handle, nullptr);
    vkFreeMemory(device.handle, memory, nullptr);
}

void Buffer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size        = size;
    bufferInfo.usage       = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(device.handle, &bufferInfo, nullptr, &handle);
    ASSERT_VULKAN(result, "Failed to create buffer");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device.handle, handle, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    result = vkAllocateMemory(device.handle, &allocInfo, nullptr, &memory);
    ASSERT_VULKAN(result, "Failed to allocate buffer memory");

    vkBindBufferMemory(device.handle, handle, memory, 0);
}

uint32_t Buffer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(device.physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;

    cerr << "failed to find suitable memory type!" << endl;
}

void Buffer::copy(Buffer src, VkDeviceSize size)
{
    CommandBuffer commandBuffer = CommandBuffer(device);
    commandBuffer.begin();
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer.handle, src.handle, handle, 1, &copyRegion);

    commandBuffer.end();

    // TODO: Find a better solution
    handle = src.handle;
    memory = src.memory;
}
