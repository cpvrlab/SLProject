#include "PhysicalDevice.h"

PhysicalDevice::PhysicalDevice(const Instance* instance, VkPhysicalDevice physicalDevice) : instance{instance}, handle{physicalDevice}
{
    vkGetPhysicalDeviceFeatures(physicalDevice, &features);
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
}

PhysicalDevice::~PhysicalDevice()
{
}
