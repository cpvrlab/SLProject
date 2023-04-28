#include "Device.h"
#include <set>

//-----------------------------------------------------------------------------
Device::Device(Instance&                 instance,
               VkPhysicalDevice&         physicalDevice,
               VkSurfaceKHR              surface,
               const vector<const char*> extensions) : _instance{instance},
                                                       _physicalDevice{physicalDevice},
                                                       _surface{surface}
{
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t>              uniqueQueueFamilies = {indices.graphicsFamily, indices.presentFamily};

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount       = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos       = queueCreateInfos.data();
    createInfo.pEnabledFeatures        = &deviceFeatures;
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
#if IS_DEBUGMODE_ON
    createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
#else
    createInfo.enabledLayerCount = 0;
#endif
    VkResult result = vkCreateDevice(physicalDevice, &createInfo, nullptr, &_handle);
    ASSERT_VULKAN(result, "Failed to create logical device");

    vkGetDeviceQueue(_handle, indices.graphicsFamily, 0, &_graphicsQueue);
    vkGetDeviceQueue(_handle, indices.presentFamily, 0, &_presentQueue);

    createCommandPool();
}
//-----------------------------------------------------------------------------
void Device::destroy()
{
    for (size_t i = 0; i < 2; i++)
    {
        if (_renderFinishedSemaphores[i] != VK_NULL_HANDLE)
            vkDestroySemaphore(_handle, _renderFinishedSemaphores[i], nullptr);
        if (_imageAvailableSemaphores[i] != VK_NULL_HANDLE)
            vkDestroySemaphore(_handle, _imageAvailableSemaphores[i], nullptr);
        if (_inFlightFences[i] != VK_NULL_HANDLE)
            vkDestroyFence(_handle, _inFlightFences[i], nullptr);
        // if (imagesInFlight[i] != VK_NULL_HANDLE)
        //     vkDestroyFence(_handle, imagesInFlight[i], nullptr);
    }

    vkDestroyCommandPool(_handle, _commandPool, nullptr);
    vkDestroyDevice(_handle, nullptr);
    vkDestroySurfaceKHR(_instance.handle, _surface, nullptr);
}
//-----------------------------------------------------------------------------
QueueFamilyIndices Device::findQueueFamilies(VkPhysicalDevice device)
{
    QueueFamilyIndices indices{};

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device,
                                             &queueFamilyCount,
                                             queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies)
    {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            indices.graphicsFamily = i;

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _surface, &presentSupport);

        if (presentSupport)
            indices.presentFamily = i;

        i++;
    }

    return indices;
}
//-----------------------------------------------------------------------------
void Device::createCommandPool()
{
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(_physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;

    VkResult result = vkCreateCommandPool(_handle, &poolInfo, nullptr, &_commandPool);
    ASSERT_VULKAN(result, "Failed to create command pool");
}
//-----------------------------------------------------------------------------
void Device::createSyncObjects(Swapchain& swapchain)
{
    _imageAvailableSemaphores.resize(2);
    _renderFinishedSemaphores.resize(2);
    _inFlightFences.resize(2);
    _imagesInFlight.resize(swapchain.images().size(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < 2; i++)
        if (vkCreateSemaphore(_handle,
                              &semaphoreInfo,
                              nullptr,
                              &_imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(_handle,
                              &semaphoreInfo,
                              nullptr,
                              &_renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(_handle,
                          &fenceInfo,
                          nullptr,
                          &_inFlightFences[i]) != VK_SUCCESS)
            std::cerr << "failed to create synchronization objects for a frame!" << std::endl;
}
//-----------------------------------------------------------------------------
void Device::waitIdle()
{
    vkDeviceWaitIdle(_handle);
}
//-----------------------------------------------------------------------------
