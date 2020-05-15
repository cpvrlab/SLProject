#pragma once

#include <vector>

#include "Instance.h"
#include "Swapchain.h"

class Swapchain;

//-----------------------------------------------------------------------------
struct QueueFamilyIndices
{
    uint32_t graphicsFamily;
    uint32_t presentFamily;
};
//-----------------------------------------------------------------------------
class Device
{
public:
    Device(Instance& instance,
           const VkPhysicalDevice&,
           VkSurfaceKHR,
           const vector<const char*> extensions);

    void               destroy();
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    void               createCommandPool();
    void               createSyncObjects(Swapchain& swapchain);
    void               waitIdle();

public:
    Instance&                     instance;
    const VkPhysicalDevice&       physicalDevice;
    VkSurfaceKHR                  surface{VK_NULL_HANDLE};
    VkDevice                      handle{VK_NULL_HANDLE};
    vector<VkExtensionProperties> device_extensions;
    vector<const char*>           enabled_extensions;
    VkQueue                       graphicsQueue;
    VkQueue                       presentQueue;
    VkCommandPool                 commandPool;
    vector<VkFence>               inFlightFences;
    vector<VkFence>               imagesInFlight;
    vector<VkSemaphore>           imageAvailableSemaphores;
    vector<VkSemaphore>           renderFinishedSemaphores;
};
//-----------------------------------------------------------------------------
