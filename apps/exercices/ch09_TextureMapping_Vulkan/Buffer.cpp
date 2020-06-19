#include "Buffer.h"

//-----------------------------------------------------------------------------
void Buffer::destroy()
{
    if (_handle != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(_device.handle(), _handle, nullptr);
        vkFreeMemory(_device.handle(), _memory, nullptr);
    }
}
//-----------------------------------------------------------------------------
void Buffer::free()
{
    vkDestroyBuffer(_device.handle(), _handle, nullptr);
    vkFreeMemory(_device.handle(), _memory, nullptr);
}
//-----------------------------------------------------------------------------
void Buffer::createBuffer(VkDeviceSize          size,
                          VkBufferUsageFlags    usage,
                          VkMemoryPropertyFlags properties)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size        = size;
    bufferInfo.usage       = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(_device.handle(), &bufferInfo, nullptr, &_handle);
    ASSERT_VULKAN(result, "Failed to create buffer");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(_device.handle(), _handle, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    result = vkAllocateMemory(_device.handle(), &allocInfo, nullptr, &_memory);
    ASSERT_VULKAN(result, "Failed to allocate buffer _memory");

    result = vkBindBufferMemory(_device.handle(), _handle, _memory, 0);
    ASSERT_VULKAN(result, "Failed to bind Buffer _memory!");
}
//-----------------------------------------------------------------------------
uint32_t Buffer::findMemoryType(uint32_t              typeFilter,
                                VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(_device.physicalDevice(), &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;

    // TODO:
    cerr << "failed to find suitable _memory type!" << endl;

    return UINT32_MAX;
}
//-----------------------------------------------------------------------------
void Buffer::copy(Buffer src, VkDeviceSize size)
{
    CommandBuffer commandBuffer = CommandBuffer(_device);
    commandBuffer.begin();
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer.handle(), src.handle(), _handle, 1, &copyRegion);

    commandBuffer.end();

    // TODO: Find a better solution
    // _handle = src._handle;
    // _memory = src._memory;
}
//-----------------------------------------------------------------------------
void Buffer::createVertexBuffer(const vector<Vertex>& vertices)
{
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    Buffer stagingBuffer = Buffer(_device);
    stagingBuffer.createBuffer(bufferSize,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(_device.handle(), stagingBuffer._memory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(_device.handle(), stagingBuffer._memory);

    createBuffer(bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copy(stagingBuffer, bufferSize);

    stagingBuffer.free();
}
//-----------------------------------------------------------------------------
void Buffer::createIndexBuffer(const vector<uint16_t> indices)
{
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    Buffer stagingBuffer = Buffer(_device);
    stagingBuffer.createBuffer(bufferSize,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(_device.handle(), stagingBuffer._memory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(_device.handle(), stagingBuffer._memory);

    createBuffer(bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copy(stagingBuffer, bufferSize);

    stagingBuffer.free();
}
//-----------------------------------------------------------------------------
