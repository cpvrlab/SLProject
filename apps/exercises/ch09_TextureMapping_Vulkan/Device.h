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
    Device(Instance&                 instance,
           VkPhysicalDevice&         physicalDevice,
           VkSurfaceKHR              surface,
           const vector<const char*> extensions);

    // Device(const Device&) = delete;
    // Device(Device&&)      = delete;
    // Device& operator=(const Device&) = delete;
    // Device& operator=(Device&&) = delete;

    void               destroy();
    void               createCommandPool();
    void               waitIdle();
    void               createSyncObjects(Swapchain& swapchain);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

    // Getter
    Instance&           instance() const { return _instance; }
    VkPhysicalDevice&   physicalDevice() const { return _physicalDevice; }
    VkSurfaceKHR        surface() const { return _surface; }
    VkDevice            handle() const { return _handle; }
    VkCommandPool       commandPool() const { return _commandPool; }
    VkQueue             graphicsQueue() const { return _graphicsQueue; }
    VkQueue             presentQueue() const { return _presentQueue; }
    vector<VkFence>     inFlightFences() const { return _inFlightFences; }
    vector<VkFence>     imagesInFlight() const { return _imagesInFlight; }
    vector<VkSemaphore> imageAvailableSemaphores() const { return _imageAvailableSemaphores; }
    vector<VkSemaphore> renderFinishedSemaphores() const { return _renderFinishedSemaphores; }

private:
    Instance&                     _instance;
    VkPhysicalDevice&             _physicalDevice;
    VkSurfaceKHR                  _surface{VK_NULL_HANDLE};
    VkDevice                      _handle{VK_NULL_HANDLE};
    vector<VkExtensionProperties> _device_extensions;
    vector<const char*>           _enabled_extensions;
    VkQueue                       _graphicsQueue;
    VkQueue                       _presentQueue;
    VkCommandPool                 _commandPool;
    vector<VkFence>               _inFlightFences;
    vector<VkFence>               _imagesInFlight;
    vector<VkSemaphore>           _imageAvailableSemaphores;
    vector<VkSemaphore>           _renderFinishedSemaphores;
};
//-----------------------------------------------------------------------------
