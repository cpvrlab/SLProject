#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include "Device.h"
#include "RenderPass.h"
#include "TextureImage.h"

#include <vector>

class Swapchain;
class TextureImage;

//-----------------------------------------------------------------------------
class Framebuffer
{
public:
    Framebuffer(Device&             device,
                const RenderPass&   renderPass,
                const Swapchain&    swapchain,
                const TextureImage& depthImage);
    void destroy();

    // Getter
    vector<VkFramebuffer> handle() const { return _handle; }

private:
    void createFramebuffer(const VkRenderPass        renderPass,
                           const VkExtent2D          swapchainExtent,
                           const vector<VkImageView> swapchainImageViews,
                           const VkImageView         depthImageView);

    Device&               _device;
    vector<VkFramebuffer> _handle{VK_NULL_HANDLE};
};
//-----------------------------------------------------------------------------
#endif
