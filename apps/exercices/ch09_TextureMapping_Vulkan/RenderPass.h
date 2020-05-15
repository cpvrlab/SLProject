#pragma once

#include "Device.h"
#include "Swapchain.h"

//-----------------------------------------------------------------------------
class RenderPass
{
public:
    RenderPass(Device& device, Swapchain& swapchain);
    void destroy();

public:
    Device&      device;
    VkRenderPass handle{VK_NULL_HANDLE};
};
//-----------------------------------------------------------------------------
