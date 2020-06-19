#pragma once

#include "Device.h"
#include "RenderPass.h"

#include <vector>

class Swapchain;

//-----------------------------------------------------------------------------
class Framebuffer
{
public:
    Framebuffer(Device&           device,
                const RenderPass& renderPass,
                const Swapchain&  swapchain);
    void destroy();

    // Getter
    vector<VkFramebuffer> handle() const { return _handle; }

private:
    void createFramebuffer(const VkRenderPass        renderPass,
                           const VkExtent2D          swapchainExtent,
                           const vector<VkImageView> swapchainImageViews);

    Device&               _device;
    vector<VkFramebuffer> _handle{VK_NULL_HANDLE};
};
//-----------------------------------------------------------------------------
