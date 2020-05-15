#include "UniformBuffer.h"

UniformBuffer::UniformBuffer(Device& device, Swapchain& swapchain, SLMat4f& camera) : device{device}, swapchain{swapchain}, camera{camera}
{
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    buffers.resize(swapchain.images.size());

    for (size_t i = 0; i < swapchain.images.size(); i++)
    {
        buffers[i] = new Buffer(device);
        buffers[i]->createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
}

void UniformBuffer::destroy()
{
    for (size_t i = 0; i < buffers.size(); i++)
        if (buffers[i] != nullptr)
            delete (buffers[i]);
}

void UniformBuffer::update(uint32_t currentImage)
{
    UniformBufferObject ubo{};
    ubo.model = SLMat4f(0.0f, 0.0f, 0.0f);
    ubo.view  = camera;
    ubo.proj.perspective(40,
                         (float)swapchain.extent.width / (float)swapchain.extent.height,
                         0.1f,
                         20.0f);

    void* data;
    vkMapMemory(device.handle, buffers[currentImage]->memory, 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device.handle, buffers[currentImage]->memory);
}
