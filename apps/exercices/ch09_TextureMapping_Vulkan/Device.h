#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>

#include "Instance.h"
#include "CommandPool.h"

struct QueueFamilyIndices
{
    uint32_t graphicsFamily;
    uint32_t presentFamily;
};

class Device
{
public:
    Device(const VkPhysicalDevice&, VkSurfaceKHR, const std::vector<const char*> extensions);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    void               createCommandPool();

public:
    const VkPhysicalDevice&            physicalDevice;
    VkSurfaceKHR                       surface{VK_NULL_HANDLE};
    VkDevice                           handle{VK_NULL_HANDLE};
    std::vector<VkExtensionProperties> device_extensions;
    std::vector<const char*>           enabled_extensions;
    VkQueue                            graphicsQueue;
    VkQueue                            presentQueue;
    VkCommandPool                      commandPool;
    std::vector<VkFence>               inFlightFence;
    std::vector<VkFence>               imageInFlight;
};
