#include "PhysicalDevice.h"

//-----------------------------------------------------------------------------
PhysicalDevice::PhysicalDevice(const Instance*  instance,
                               VkPhysicalDevice physicalDevice) : _instance{instance},
                                                                  _handle{physicalDevice}
{
    vkGetPhysicalDeviceFeatures(physicalDevice, &_features);
    vkGetPhysicalDeviceProperties(physicalDevice, &_properties);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &_memoryProperties);
}
//-----------------------------------------------------------------------------
