#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <set>

#define IS_DEBUGMODE_ON true

struct QueueFamilyIndices
{
    uint32_t graphicsFamily;
    uint32_t presentFamily;
};

struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

class TextureMapping
{
public:
    const int                      WIDTH                = 800;
    const int                      HEIGHT               = 600;
    const int                      MAX_FRAMES_IN_FLIGHT = 2;
    const std::vector<const char*> validationLayers     = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*> deviceExtensions     = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

private:
    GLFWwindow*                  window;
    VkInstance                   instance;
    VkDebugUtilsMessengerEXT     debugMessenger;
    VkSurfaceKHR                 surface;
    VkPhysicalDevice             physicalDevice = VK_NULL_HANDLE;
    VkDevice                     device;
    VkQueue                      graphicsQueue;
    VkQueue                      presentQueue;
    VkSwapchainKHR               swapchain;
    std::vector<VkImage>         swapchainImages;
    VkFormat                     swapchainImageFormat;
    VkExtent2D                   swapchainExtent;
    std::vector<VkImageView>     swapchainImageViews;
    std::vector<VkFramebuffer>   swapchainFramebuffers;
    VkRenderPass                 renderPass;
    VkPipelineLayout             pipelineLayout;
    VkPipeline                   graphicsPipeline;
    VkCommandPool                commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore>     imageAvailableSemaphores;
    std::vector<VkSemaphore>     renderFinishedSemaphores;
    std::vector<VkFence>         inFlightFences;
    std::vector<VkFence>         imagesInFlight;
    size_t                       currentFrame = 0;
    bool                         framebufferResized = false;

public:
    void run();

private:
    void initVulkan();
    void initWindow();
    void mainLoop();
    void cleanupSwapchain();
    void cleanup();
    void recreateSwapchain();
    void createInstance();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT&);
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void createRenderPass();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();
    void drawFrame();
    void DestroyDebugUtilsMessengerEXT(VkInstance, 
                                       VkDebugUtilsMessengerEXT, 
                                       const VkAllocationCallbacks*);
    bool checkValidationLayerSupport();
    bool isDeviceSuitable(VkPhysicalDevice);
    bool checkDeviceExtensionSupport(VkPhysicalDevice);
    VkShaderModule           createShaderModule(const std::vector<char>&);
    VkSurfaceFormatKHR       chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>&);
    VkPresentModeKHR         chooseSwapPresentMode(const std::vector<VkPresentModeKHR>&);     // Must be replaced
    VkExtent2D               chooseSwapExtent(const VkSurfaceCapabilitiesKHR&);
    SwapchainSupportDetails  querySwapchainSupport(VkPhysicalDevice);
    QueueFamilyIndices       findQueueFamilies(VkPhysicalDevice);
    std::vector<const char*> getRequiredExtensions();
    VkResult CreateDebugUtilsMessengerEXT(VkInstance,
                                          const VkDebugUtilsMessengerCreateInfoEXT*, 
                                          const VkAllocationCallbacks*, 
                                          VkDebugUtilsMessengerEXT*);
    
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        auto app                = reinterpret_cast<TextureMapping*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL TextureMapping::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                                        VkDebugUtilsMessageTypeFlagsEXT messageType, 
                                                                        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, 
                                                                        void* pUserData)
    {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
};

static std::vector<char> readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
        throw std::runtime_error("failed to open file!");

    size_t            fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}