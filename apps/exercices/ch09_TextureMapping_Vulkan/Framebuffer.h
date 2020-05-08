#pragma once

#include "Device.h"
#include "RenderPass.h"
#include "Swapchain.h"
#include <vector>

class Framebuffer
{
public:
    Framebuffer(Device& device, const RenderPass renderPass, const Swapchain swapchain);

private:
    void createFramebuffer(const VkRenderPass renderPass, const VkExtent2D swapchainExtent, const std::vector<VkImageView> swapchainImageViews);

public:
    Device&                    device;
    std::vector<VkFramebuffer> handle{VK_NULL_HANDLE};
};
