#ifndef INSTANCE_H
#define INSTANCE_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <Utils.h>
#include <iostream>

#define VK_DEBUG
#define ASSERT_VULKAN(result, msg) \
    if (result != VK_SUCCESS) \
    Utils::exitMsg("Vulkan", msg, __LINE__, __FILE__)

// forward declare
class PhysicalDevice;

//-----------------------------------------------------------------------------
class Instance
{
public:
    Instance(const char*                applicationName,
             const vector<const char*>& requiredExtensions,
             const vector<const char*>& requiredValidationLayer);

    void destroy();

private:
    void                findSuitableGPU();
    bool                checkValidationLayerSupport(const vector<const char*>&);
    vector<const char*> getRequiredExtensions();

#if defined(VK_DEBUG)
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT&);

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                                        VkDebugUtilsMessageTypeFlagsEXT             messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        void*                                       pUserData)
    {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    VkResult CreateDebugUtilsMessengerEXT(VkInstance,
                                          const VkDebugUtilsMessengerCreateInfoEXT*,
                                          const VkAllocationCallbacks*,
                                          VkDebugUtilsMessengerEXT*);
    void     setupDebugMessenger();

    VkDebugUtilsMessengerEXT debugUtilsMessenger{VK_NULL_HANDLE};
    VkDebugReportCallbackEXT debugReportCallback{VK_NULL_HANDLE};
#endif

public:
    VkInstance          handle{VK_NULL_HANDLE};
    vector<const char*> enabled_extensions;
    VkPhysicalDevice    physicalDevice;
};
//-----------------------------------------------------------------------------
#endif
