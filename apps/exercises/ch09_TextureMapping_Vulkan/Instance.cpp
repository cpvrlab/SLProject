#include "Instance.h"
#include <cstring>

//-----------------------------------------------------------------------------
Instance::Instance(const char*                applicationName,
                   const vector<const char*>& requiredExtensions,
                   const vector<const char*>& validationLayer)
{
#if defined(VK_DEBUG)
    if (!checkValidationLayerSupport(validationLayer))
        std::cerr << "Validation layers requested, but not availale!" << std::endl;
#endif
    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = applicationName;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName        = "SL_Project";
    appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    // Extensions that should be enabled
    vector<const char*> extensions     = getRequiredExtensions();
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

#if defined(VK_DEBUG)
    createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayer.size());
    createInfo.ppEnabledLayerNames = validationLayer.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
    populateDebugMessengerCreateInfo(debugCreateInfo);
    createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
#else
    createInfo.enabledLayerCount = 0;
    createInfo.pNext             = nullptr;
#endif
    // Creates an instance of the program by the given parameter. The instance is stored in the last parameter
    VkResult result = vkCreateInstance(&createInfo, nullptr, &handle);
    ASSERT_VULKAN(result, "Failed to create instance");

    findSuitableGPU();
    setupDebugMessenger();

    // TODO: Remove later
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    std::cout << "Max Memory Allocation Count: " << properties.limits.maxMemoryAllocationCount << std::endl;
}
//-----------------------------------------------------------------------------
void Instance::destroy()
{
#if defined(VK_DEBUG)
    // if (debugUtilsMessenger != VK_NULL_HANDLE)
    //     vkDestroyDebugUtilsMessengerEXT(handle, debugUtilsMessenger, nullptr);
    // if (debugReportCallback != VK_NULL_HANDLE)
    //     vkDestroyDebugReportCallbackEXT(handle, debugReportCallback, nullptr);
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(handle,
                                                                           "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
        func(handle, debugUtilsMessenger, nullptr);
#endif

    if (handle != VK_NULL_HANDLE)
        vkDestroyInstance(handle, nullptr);
}
//-----------------------------------------------------------------------------
void Instance::findSuitableGPU()
{
    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(handle, &physicalDeviceCount, nullptr);

    if (physicalDeviceCount < 1)
        std::cerr << "Could not find a physical device that supports Vulkan!" << std::endl;

    vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(handle, &physicalDeviceCount, physicalDevices.data());
    physicalDevice = physicalDevices[0];
}
//-----------------------------------------------------------------------------
bool Instance::checkValidationLayerSupport(const vector<const char*>& validationLayers)
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers)
    {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers)
            if (strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }

        if (!layerFound)
            return false;
    }

    return true;
}
//-----------------------------------------------------------------------------
vector<const char*> Instance::getRequiredExtensions()
{
    uint32_t     glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    vector<const char*> extensions(glfwExtensions,
                                   glfwExtensions + glfwExtensionCount);

#if defined(VK_DEBUG)
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
    return extensions;
}
//-----------------------------------------------------------------------------
#if defined(VK_DEBUG)
void Instance::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
    createInfo                 = {};
    createInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}
//-----------------------------------------------------------------------------
VkResult Instance::CreateDebugUtilsMessengerEXT(VkInstance                                instance,
                                                const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                                const VkAllocationCallbacks*              pAllocator,
                                                VkDebugUtilsMessengerEXT*                 pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,
                                                                          "vkCreateDebugUtilsMessengerEXT");

    if (func != nullptr)
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    else
        return VK_ERROR_EXTENSION_NOT_PRESENT;
}
//-----------------------------------------------------------------------------
void Instance::setupDebugMessenger()
{
#    if !defined(VK_DEBUG)
    return;
#    endif
    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    VkResult result = CreateDebugUtilsMessengerEXT(handle,
                                                   &createInfo,
                                                   nullptr,
                                                   &debugUtilsMessenger);
    ASSERT_VULKAN(result, "Failed to set up debug messenger");
}
#endif
//-----------------------------------------------------------------------------
