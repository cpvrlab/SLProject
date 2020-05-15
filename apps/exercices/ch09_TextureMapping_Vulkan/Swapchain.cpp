
#include "Swapchain.h"
#include "GLFW/glfw3.h"
#include <algorithm>

//-----------------------------------------------------------------------------
Swapchain::Swapchain(Device&     device,
                     GLFWwindow* window) : device{device}
{
    swapchainSupport = querySwapchainSupport(device);
    surfaceFormat    = chooseSwapSurfaceFormat(swapchainSupport.formats);
    presentMode      = chooseSwapPresentMode(swapchainSupport.presentModes);
    extent           = chooseSwapExtent(swapchainSupport.capabilities, window);

    uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;
    if ((swapchainSupport.capabilities.maxImageCount > 0) &&
        (imageCount > swapchainSupport.capabilities.maxImageCount))
        imageCount = swapchainSupport.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface          = device.surface;
    createInfo.minImageCount    = imageCount;
    createInfo.imageFormat      = surfaceFormat.format;
    createInfo.imageColorSpace  = surfaceFormat.colorSpace;
    createInfo.imageExtent      = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices              = device.findQueueFamilies(device.physicalDevice);
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

    createInfo.preTransform   = swapchainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode    = presentMode;
    createInfo.clipped        = VK_TRUE;

    VkResult result = vkCreateSwapchainKHR(device.handle, &createInfo, nullptr, &handle);
    ASSERT_VULKAN(result, "Failed to create swapchain");

    vkGetSwapchainImagesKHR(device.handle, handle, &imageCount, nullptr);
    images.resize(imageCount);
    vkGetSwapchainImagesKHR(device.handle, handle, &imageCount, images.data());

    createImageViews();
}
//-----------------------------------------------------------------------------
void Swapchain::destroy()
{
    for (size_t i = 0; i < imageViews.size(); i++)
        if (imageViews[i] != VK_NULL_HANDLE)
            vkDestroyImageView(device.handle, imageViews[i], nullptr);

    if (handle != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(device.handle, handle, nullptr);
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
        // VK_PRESENT_MODE_MAILBOX_KHR: The created images are redrawn when the queue is full
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

        actualExtent.width  = max(capabilities.minImageExtent.width,
                                 min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = max(capabilities.minImageExtent.height,
                                  min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}
//-----------------------------------------------------------------------------
SwapchainSupportDetails Swapchain::querySwapchainSupport(Device& device)
{
    SwapchainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.physicalDevice,
                                              device.surface,
                                              &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device.physicalDevice,
                                         device.surface,
                                         &formatCount,
                                         nullptr);

    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device.physicalDevice,
                                             device.surface,
                                             &formatCount,
                                             details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device.physicalDevice,
                                              device.surface,
                                              &presentModeCount,
                                              nullptr);

    if (presentModeCount != 0)
    {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device.physicalDevice,
                                                  device.surface,
                                                  &presentModeCount,
                                                  details.presentModes.data());
    }

    return details;
}
//-----------------------------------------------------------------------------
void Swapchain::createImageViews()
{
    imageViews.resize(images.size());

    for (size_t i = 0; i < images.size(); i++)
        imageViews[i] = createImageView(images[i], surfaceFormat.format);
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

    VkResult result = vkCreateImageView(device.handle, &viewInfo, nullptr, &imageView);
    ASSERT_VULKAN(result, "Failed to create texture image view!");

    return imageView;
}
//-----------------------------------------------------------------------------
