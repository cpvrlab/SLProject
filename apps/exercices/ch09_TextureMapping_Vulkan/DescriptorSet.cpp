#include "DescriptorSet.h"

#include "UniformBufferObject.h"

DescriptorSet::DescriptorSet(Device& device, Swapchain& swapchain, DescriptorSetLayout& descriptorSetLayout, DescriptorPool& descriptorPool, UniformBuffer& uniformBuffer, Sampler& textureSampler, TextureImage& textureImage) : _device{device}
{
    vector<VkDescriptorSetLayout> layouts(swapchain.images().size(), descriptorSetLayout.handle());
    VkDescriptorSetAllocateInfo   allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = descriptorPool.handle();
    allocInfo.descriptorSetCount = static_cast<uint32_t>(swapchain.images().size());
    allocInfo.pSetLayouts        = layouts.data();

    _handles.resize(swapchain.images().size());
    VkResult result = vkAllocateDescriptorSets(_device.handle(), &allocInfo, _handles.data());
    ASSERT_VULKAN(result, "Failed to allocate descriptor sets");

    for (size_t i = 0; i < swapchain.images().size(); i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffer.buffers()[i]->handle();
        bufferInfo.offset = 0;
        bufferInfo.range  = sizeof(UniformBufferObject);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView   = textureImage.imageView();
        imageInfo.sampler     = textureSampler.handle();

        array<VkWriteDescriptorSet, 2> descriptorWrites{};

        descriptorWrites[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet          = _handles[i];
        descriptorWrites[0].dstBinding      = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo     = &bufferInfo;

        descriptorWrites[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet          = _handles[i];
        descriptorWrites[1].dstBinding      = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo      = &imageInfo;

        vkUpdateDescriptorSets(device.handle(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}
