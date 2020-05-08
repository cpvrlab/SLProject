#include "Framebuffer.h"

Framebuffer::Framebuffer(Device& device, const RenderPass renderPass, const Swapchain swapchain) : device{device}
{
    createFramebuffer(renderPass.handle, swapchain.extent, swapchain.imageViews);
}

void Framebuffer::createFramebuffer(const VkRenderPass renderPass, const VkExtent2D swapchainExtent, const std::vector<VkImageView> swapchainImageViews)
{
    handle.resize(swapchainImageViews.size());

    for (size_t i = 0; i < swapchainImageViews.size(); i++)
    {
        VkImageView attachments[] = {swapchainImageViews[i]};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass      = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments    = attachments;
        framebufferInfo.width           = swapchainExtent.width;
        framebufferInfo.height          = swapchainExtent.height;
        framebufferInfo.layers          = 1;

        VkResult result = vkCreateFramebuffer(device.handle, &framebufferInfo, nullptr, &handle[i]);
        ASSERT_VULKAN(result, "Failed to create framebuffer");
    }
}
