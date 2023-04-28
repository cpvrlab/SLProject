#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <Utils.h>
#include <iostream>

#include "Device.h"

//-----------------------------------------------------------------------------
struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR   capabilities;
    vector<VkSurfaceFormatKHR> formats;
    vector<VkPresentModeKHR>   presentModes;
};
//-----------------------------------------------------------------------------
class Device;
//-----------------------------------------------------------------------------
class Swapchain
{
public:
    Swapchain(Device&     device,
              GLFWwindow* window);
    void destroy();

    // Getter
    VkSwapchainKHR      handle() const { return _handle; }
    VkExtent2D          extent() const { return _extent; }
    vector<VkImage>     images() const { return _images; }
    vector<VkImageView> imageViews() const { return _imageViews; }
    VkSurfaceFormatKHR  surfaceFormat() const { return _surfaceFormat; }

private:
    VkSurfaceFormatKHR      chooseSwapSurfaceFormat(const vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR        chooseSwapPresentMode(const vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D              chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                             GLFWwindow*                     window);
    SwapchainSupportDetails querySwapchainSupport(Device& device);
    void                    createImageViews();
    VkImageView             createImageView(VkImage image, VkFormat format);

    Device&                 _device;
    VkSwapchainKHR          _handle{VK_NULL_HANDLE};
    VkExtent2D              _extent;
    vector<VkImage>         _images;
    VkSurfaceFormatKHR      _surfaceFormat;
    VkPresentModeKHR        _presentMode;
    SwapchainSupportDetails _swapchainSupport;
    vector<VkImageView>     _imageViews;
};
//-----------------------------------------------------------------------------
