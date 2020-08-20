#pragma once

#include "Instance.h"

//-----------------------------------------------------------------------------
class PhysicalDevice
{
public:
    PhysicalDevice(const Instance*  instance,
                   VkPhysicalDevice physicalDevice);
    void destroy();

    // Getter
    VkPhysicalDevice handle() const { return _handle; }

private:
    const Instance*                  _instance;
    VkPhysicalDevice                 _handle{VK_NULL_HANDLE};
    VkPhysicalDeviceFeatures         _features{};
    VkPhysicalDeviceProperties       _properties;
    VkPhysicalDeviceMemoryProperties _memoryProperties;
};
//-----------------------------------------------------------------------------
