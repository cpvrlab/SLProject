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

private:
    void createFramebuffer(const VkRenderPass        renderPass,
                           const VkExtent2D          swapchainExtent,
                           const vector<VkImageView> swapchainImageViews);

public:
    Device&               device;
    vector<VkFramebuffer> handle{VK_NULL_HANDLE};
};
//-----------------------------------------------------------------------------
