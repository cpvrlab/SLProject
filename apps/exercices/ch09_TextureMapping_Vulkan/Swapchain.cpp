#include "Swapchain.h"

#include "GLFW/glfw3.h"
#include <algorithm>

//-----------------------------------------------------------------------------
Swapchain::Swapchain(Device&     _device,
                     GLFWwindow* window) : _device{_device}
{
    _swapchainSupport = querySwapchainSupport(_device);
    _surfaceFormat    = chooseSwapSurfaceFormat(_swapchainSupport.formats);
    _presentMode      = chooseSwapPresentMode(_swapchainSupport.presentModes);
    _extent           = chooseSwapExtent(_swapchainSupport.capabilities, window);

    uint32_t imageCount = _swapchainSupport.capabilities.minImageCount + 1;
    if ((_swapchainSupport.capabilities.maxImageCount > 0) &&
        (imageCount > _swapchainSupport.capabilities.maxImageCount))
        imageCount = _swapchainSupport.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface          = _device.surface();
    createInfo.minImageCount    = imageCount;
    createInfo.imageFormat      = _surfaceFormat.format;
    createInfo.imageColorSpace  = _surfaceFormat.colorSpace;
    createInfo.imageExtent      = _extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices              = _device.findQueueFamilies(_device.physicalDevice());
    uint32_t           queueFamilyIndices[] = {indices.graphicsFamily, indices.presentFamily};

    if (indices.graphicsFamily != indices.presentFamily)
    {
        createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices   = queueFamilyIndices;
    }
    else
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform   = _swapchainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode    = _presentMode;
    createInfo.clipped        = VK_TRUE;

    VkResult result = vkCreateSwapchainKHR(_device.handle(), &createInfo, nullptr, &_handle);
    ASSERT_VULKAN(result, "Failed to create swapchain");

    vkGetSwapchainImagesKHR(_device.handle(), _handle, &imageCount, nullptr);
    _images.resize(imageCount);
    vkGetSwapchainImagesKHR(_device.handle(), _handle, &imageCount, _images.data());

    createImageViews();
}
//-----------------------------------------------------------------------------
void Swapchain::destroy()
{
    for (size_t i = 0; i < _imageViews.size(); i++)
        if (_imageViews[i] != VK_NULL_HANDLE)
            vkDestroyImageView(_device.handle(), _imageViews[i], nullptr);

    if (_handle != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(_device.handle(), _handle, nullptr);
}
//-----------------------------------------------------------------------------
VkSurfaceFormatKHR Swapchain::chooseSwapSurfaceFormat(const vector<VkSurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats)
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return availableFormat;

    return availableFormats[0];
}
//-----------------------------------------------------------------------------
VkPresentModeKHR Swapchain::chooseSwapPresentMode(const vector<VkPresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes)
        // VK_PRESENT_MODE_MAILBOX_KHR: The created _images are redrawn when the queue is full
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            return availablePresentMode;
    // VK_PRESENT_MODE_FIFO_KHR: First in first out. If the queue is full, the program waits
    return VK_PRESENT_MODE_FIFO_KHR;
}
//-----------------------------------------------------------------------------
VkExtent2D Swapchain::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                       GLFWwindow*                     window)
{
    if (capabilities.currentExtent.width != UINT32_MAX)
        return capabilities.currentExtent;
    else
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height)};

        actualExtent.width  = std::max(capabilities.minImageExtent.width,
                                      std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height,
                                       std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}
//-----------------------------------------------------------------------------
SwapchainSupportDetails Swapchain::querySwapchainSupport(Device& _device)
{
    SwapchainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_device.physicalDevice(),
                                              _device.surface(),
                                              &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(_device.physicalDevice(),
                                         _device.surface(),
                                         &formatCount,
                                         nullptr);

    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(_device.physicalDevice(),
                                             _device.surface(),
                                             &formatCount,
                                             details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(_device.physicalDevice(),
                                              _device.surface(),
                                              &presentModeCount,
                                              nullptr);

    if (presentModeCount != 0)
    {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(_device.physicalDevice(),
                                                  _device.surface(),
                                                  &presentModeCount,
                                                  details.presentModes.data());
    }

    return details;
}
//-----------------------------------------------------------------------------
void Swapchain::createImageViews()
{
    _imageViews.resize(_images.size());

    for (size_t i = 0; i < _images.size(); i++)
        _imageViews[i] = createImageView(_images[i], _surfaceFormat.format);
}
//-----------------------------------------------------------------------------
VkImageView Swapchain::createImageView(VkImage image, VkFormat format)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = image;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = format;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    VkImageView imageView;

    VkResult result = vkCreateImageView(_device.handle(), &viewInfo, nullptr, &imageView);
    ASSERT_VULKAN(result, "Failed to create texture image view!");

    return imageView;
}
//-----------------------------------------------------------------------------
