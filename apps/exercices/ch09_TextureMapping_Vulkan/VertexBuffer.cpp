/*
#include "VertexBuffer.h"

VertexBuffer::VertexBuffer(Device& device, const vector<Vertex>& vertices) : device{device}
{
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    Buffer stagingBuffer = Buffer(device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data;
    vkMapMemory(device.handle, stagingBuffer.memory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device.handle, stagingBuffer.memory);

    Buffer vBuffer = Buffer(device, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // ERROR: New and better solution bc it doesnt copy to the right buffer (this is not flaged as VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    copy(stagingBuffer, bufferSize);

    vkDestroyBuffer(device.handle, stagingBuffer.handle, nullptr);
    vkFreeMemory(device.handle, stagingBuffer.memory, nullptr);
}

void VertexBuffer::copy(Buffer src, VkDeviceSize size)
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
*/
