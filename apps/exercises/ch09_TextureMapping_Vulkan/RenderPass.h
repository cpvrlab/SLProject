#pragma once

#include "Device.h"
#include "Swapchain.h"

//-----------------------------------------------------------------------------
class RenderPass
{
public:
    RenderPass(Device& device, Swapchain& swapchain);
    void destroy();

    // Getter
    VkRenderPass handle() const { return _handle; }

private:
    Device&      _device;
    VkRenderPass _handle{VK_NULL_HANDLE};
};
//-----------------------------------------------------------------------------
