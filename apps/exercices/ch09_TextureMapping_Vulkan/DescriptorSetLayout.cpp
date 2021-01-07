#include "DescriptorSetLayout.h"
#include <array>

//-----------------------------------------------------------------------------
DescriptorSetLayout::DescriptorSetLayout(Device& device) : _device{device}
{
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding            = 0;
    uboLayoutBinding.descriptorCount    = 1;
    uboLayoutBinding.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    uboLayoutBinding.stageFlags         = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding            = 1;
    samplerLayoutBinding.descriptorCount    = 1;
    samplerLayoutBinding.descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
    VkDescriptorSetLayoutCreateInfo             layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings    = bindings.data();

    VkResult result = vkCreateDescriptorSetLayout(device.handle(),
                                                  &layoutInfo,
                                                  nullptr,
                                                  &_handle);
    ASSERT_VULKAN(result, "Failed to create descriptor set layout");
}
//-----------------------------------------------------------------------------
void DescriptorSetLayout::destroy()
{
    if (_handle != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(_device.handle(), _handle, nullptr);
}
//-----------------------------------------------------------------------------
