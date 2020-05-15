#pragma once

#include "Instance.h"

class PhysicalDevice
{
public:
    PhysicalDevice(const Instance* instance, VkPhysicalDevice physicalDevice);
    void destroy();

public:
    const Instance*                  instance;
    VkPhysicalDevice                 handle{VK_NULL_HANDLE};
    VkPhysicalDeviceFeatures         features{};
    VkPhysicalDeviceProperties       properties;
    VkPhysicalDeviceMemoryProperties memoryProperties;
};
