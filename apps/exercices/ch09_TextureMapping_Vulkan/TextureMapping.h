#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GL/glew.h>    // OpenGL headers
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
#include <array>
#include <glUtils.h>
#include <Utils.h>
#include <math/SLVec2.h>
#include <math/SLVec3.h>
#include <chrono>
#include <Utils.h>
#include <SL.h>         // Basic SL type definitions
#include <CVImage.h>    // Image class for image loading
#include <glUtils.h> 

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

struct UniformBufferObject
{
    SLMat4f model;
    SLMat4f view;
    SLMat4f proj;
};

struct Vertex
{
    SLVec2f pos;
    SLVec3f color;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding                         = 0;
        bindingDescription.stride                          = sizeof(Vertex);
        bindingDescription.inputRate                       = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};

        attributeDescriptions[0].binding  = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format   = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset   = offsetof(Vertex, pos);

        attributeDescriptions[1].binding  = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset   = offsetof(Vertex, color);

        return attributeDescriptions;
    }
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
    VkBuffer                     vertexBuffer;
    VkDeviceMemory               vertexBufferMemory;
    VkSwapchainKHR               swapchain;
    std::vector<VkImage>         swapchainImages;
    VkFormat                     swapchainImageFormat;
    VkExtent2D                   swapchainExtent;
    std::vector<VkImageView>     swapchainImageViews;
    std::vector<VkFramebuffer>   swapchainFramebuffers;
    VkRenderPass                 renderPass;
    VkDescriptorSetLayout        descriptorSetLayout;
    VkDescriptorPool             descriptorPool;
    VkPipelineLayout             pipelineLayout;
    VkPipeline                   graphicsPipeline;
    VkCommandPool                commandPool;
    VkBuffer                     stagingBuffer;
    VkDeviceMemory               stagingBufferMemory;
    VkImage                      textureImage;
    VkDeviceMemory               textureImageMemory;
    VkBuffer                     indexBuffer;
    VkDeviceMemory               indexBufferMemory;
    std::vector<VkDescriptorSet> descriptorSets;
    std::vector<VkBuffer>        uniformBuffers;
    std::vector<VkDeviceMemory>  uniformBuffersMemory;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore>     imageAvailableSemaphores;
    std::vector<VkSemaphore>     renderFinishedSemaphores;
    std::vector<VkFence>         inFlightFences;
    std::vector<VkFence>         imagesInFlight;
    size_t                       currentFrame = 0;
    bool                         framebufferResized = false;
    const std::vector<Vertex>    vertices           = { {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                                                        {{ 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
                                                        {{ 0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
                                                        {{-0.5f,  0.5f}, {1.0f, 1.0f, 1.0f}} };
    const std::vector<uint16_t> indices = { 0, 1, 2, 2, 3, 0 };

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
    void createDescriptiorSetLayout();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createDescriptorSetLayout();
    void createCommandPool();
    void createTextureImage();
    void createImage(uint32_t, uint32_t, VkFormat, VkImageTiling, VkImageUsageFlags, VkMemoryPropertyFlags, VkImage&, VkDeviceMemory&);
    void transitionImageLayout(VkImage, VkFormat, VkImageLayout, VkImageLayout);
    void copyBufferToImage(VkBuffer, VkImage, uint32_t, uint32_t);
    void createVertexBuffer();
    void createIndexBuffer();
    void createBuffer(VkDeviceSize, VkBufferUsageFlags, VkMemoryPropertyFlags, VkBuffer&, VkDeviceMemory&);
    void copyBuffer(VkBuffer, VkBuffer, VkDeviceSize);
    void updateUniformBuffer(uint32_t);
    void createCommandBuffers();
    void createSyncObjects();
    void drawFrame();
    void DestroyDebugUtilsMessengerEXT(VkInstance, 
                                       VkDebugUtilsMessengerEXT, 
                                       const VkAllocationCallbacks*);
    bool checkValidationLayerSupport();
    bool isDeviceSuitable(VkPhysicalDevice);
    bool checkDeviceExtensionSupport(VkPhysicalDevice);
    uint32_t                 findMemoryType(uint32_t, VkMemoryPropertyFlags);
    VkCommandBuffer          beginSingleTimeCommands();
    void                     endSingleTimeCommands(VkCommandBuffer);
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