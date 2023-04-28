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
    allocInfo.memoryTypeIndex = findMemoryType(_device, memRequirements.memoryTypeBits, properties);

    result = vkAllocateMemory(_device.handle(), &allocInfo, nullptr, &_memory);
    ASSERT_VULKAN(result, "Failed to allocate buffer memory");

    result = vkBindBufferMemory(_device.handle(), _handle, _memory, 0);
    ASSERT_VULKAN(result, "Failed to bind Buffer memory!");
}
//-----------------------------------------------------------------------------
uint32_t Buffer::findMemoryType(Device&               device,
                                uint32_t              typeFilter,
                                VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(device.physicalDevice(), &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;

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

    stagingBuffer.destroy();
}
//-----------------------------------------------------------------------------
void Buffer::createVertexBuffer(const SLVVec3f pos, const SLVVec3f norm, const SLVVec2f texCoord, const SLVCol4f color, const size_t size)
{
    VkDeviceSize totalBufferSize = (sizeof(SLVec3f) * 2 + sizeof(SLVec2f) + sizeof(SLCol4f)) * size;
    Buffer       stagingBuffer   = Buffer(_device);
    stagingBuffer.createBuffer(totalBufferSize,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(_device.handle(), stagingBuffer._memory, 0, totalBufferSize, 0, &data);
    char* temp = (char*)data;
    for (size_t i = 0; i < size; i++)
    {
        memcpy((temp), &pos[i], sizeof(SLVec3f));
        memcpy((temp += sizeof(SLVec3f)), &norm[i], sizeof(SLVec3f));
        memcpy((temp += sizeof(SLVec3f)), &texCoord[i], sizeof(SLVec2f));
        memcpy((temp += sizeof(SLVec2f)), &color[i], sizeof(SLCol4f));
        temp += sizeof(SLCol4f);
    }
    temp -= totalBufferSize;

    vkUnmapMemory(_device.handle(), stagingBuffer._memory);

    createBuffer(totalBufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    copy(stagingBuffer, totalBufferSize);

    stagingBuffer.destroy();
}
//-----------------------------------------------------------------------------
void Buffer::createIndexBuffer(const SLVuint indices)
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

    stagingBuffer.destroy();
}
//-----------------------------------------------------------------------------
