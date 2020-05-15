#include "Buffer.h"

void Buffer::destroy()
{
    if (handle != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device.handle, handle, nullptr);
        vkFreeMemory(device.handle, memory, nullptr);
    }
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

    result = vkBindBufferMemory(device.handle, handle, memory, 0);
    ASSERT_VULKAN(result, "Failed to bind Buffer memory!");
}

uint32_t Buffer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(device.physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;

    // TODO:
    cerr << "failed to find suitable memory type!" << endl;

    return UINT32_MAX;
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
    // handle = src.handle;
    // memory = src.memory;
}

void Buffer::createVertexBuffer(const vector<Vertex>& vertices)
{
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    Buffer stagingBuffer = Buffer(device);
    stagingBuffer.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(device.handle, stagingBuffer.memory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device.handle, stagingBuffer.memory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copy(stagingBuffer, bufferSize);

    stagingBuffer.free();
}

void Buffer::createIndexBuffer(const std::vector<uint16_t> indices)
{
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    Buffer stagingBuffer = Buffer(device);
    stagingBuffer.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(device.handle, stagingBuffer.memory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(device.handle, stagingBuffer.memory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copy(stagingBuffer, bufferSize);

    stagingBuffer.free();
}
