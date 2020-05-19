#include "DescriptorPool.h"

//-----------------------------------------------------------------------------
DescriptorPool::DescriptorPool(Device& device, Swapchain& swapchain) : device{device}
{
    array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(swapchain.images.size());
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(swapchain.images.size());

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();
    poolInfo.maxSets       = static_cast<uint32_t>(swapchain.images.size());

    VkResult result = vkCreateDescriptorPool(device.handle, &poolInfo, nullptr, &handle);
    ASSERT_VULKAN(result, "Failed to create descriptor pool");
}
//-----------------------------------------------------------------------------
void DescriptorPool::destroy()
{
    if (handle != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device.handle, handle, nullptr);
}
//-----------------------------------------------------------------------------