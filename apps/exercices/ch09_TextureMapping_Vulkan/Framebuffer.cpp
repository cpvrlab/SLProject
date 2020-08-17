#include "Framebuffer.h"

#include <array>

//-----------------------------------------------------------------------------
Framebuffer::Framebuffer(Device&             device,
                         const RenderPass&   renderPass,
                         const Swapchain&    swapchain,
                         const TextureImage& depthImage) : _device{device}
{
    createFramebuffer(renderPass.handle(),
                      swapchain.extent(),
                      swapchain.imageViews(),
                      depthImage.imageView());
}
//-----------------------------------------------------------------------------
void Framebuffer::destroy()
{
    for (size_t i = 0; i < _handle.size(); i++)
        if (_handle[i] != VK_NULL_HANDLE)
            vkDestroyFramebuffer(_device.handle(), _handle[i], nullptr);
}
//-----------------------------------------------------------------------------
void Framebuffer::createFramebuffer(const VkRenderPass        renderPass,
                                    const VkExtent2D          swapchainExtent,
                                    const vector<VkImageView> swapchainImageViews,
                                    const VkImageView         depthImageView)
{
    _handle.resize(swapchainImageViews.size());

    for (size_t i = 0; i < swapchainImageViews.size(); i++)
    {
        array<VkImageView, 2> attachments = {swapchainImageViews[i], depthImageView};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass      = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments    = attachments.data();
        framebufferInfo.width           = swapchainExtent.width;
        framebufferInfo.height          = swapchainExtent.height;
        framebufferInfo.layers          = 1;

        VkResult result = vkCreateFramebuffer(_device.handle(), &framebufferInfo, nullptr, &_handle[i]);
        ASSERT_VULKAN(result, "Failed to create framebuffer");
    }
}
//-----------------------------------------------------------------------------
