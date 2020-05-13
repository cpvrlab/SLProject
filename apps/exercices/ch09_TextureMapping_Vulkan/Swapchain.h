#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <Utils.h>
#include <iostream>

#include "Device.h"

struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR   capabilities;
    vector<VkSurfaceFormatKHR> formats;
    vector<VkPresentModeKHR>   presentModes;
};

class Device;

class Swapchain
{
public:
    Swapchain(Device&     device,
              GLFWwindow* window);

private:
    VkSurfaceFormatKHR      chooseSwapSurfaceFormat(const vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR        chooseSwapPresentMode(const vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D              chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                             GLFWwindow*                     window);
    SwapchainSupportDetails querySwapchainSupport(Device& device);
    void                    createImageViews();
    VkImageView             createImageView(VkImage image, VkFormat format);

public:
    Device&                  device;
    VkSwapchainKHR           handle{VK_NULL_HANDLE};
    VkExtent2D               extent;
    std::vector<VkImage>     images;
    VkSurfaceFormatKHR       surfaceFormat;
    VkPresentModeKHR         presentMode;
    SwapchainSupportDetails  swapchainSupport;
    std::vector<VkImageView> imageViews;
};
