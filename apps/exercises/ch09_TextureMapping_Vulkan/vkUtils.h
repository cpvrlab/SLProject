#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GL/gl3w.h> // OpenGL headers
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <set>
#include <array>
#include <glUtils.h>
#include <Utils.h>
#include <math/SLVec3.h>
#include "Vertex.cpp"

#define ASSERT_VULKAN(result, msg) \
    if (result != VK_SUCCESS) \
        Utils::exitMsg("Vulkan", msg, __LINE__, __FILE__);

#define IS_DEBUGMODE_ON true

//-----------------------------------------------------------------------------
struct QueueFamilyIndices
{
    uint32_t graphicsFamily;
    uint32_t presentFamily;
};
//-----------------------------------------------------------------------------
struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR   capabilities;
    vector<VkSurfaceFormatKHR> formats;
    vector<VkPresentModeKHR>   presentModes;
};
//-----------------------------------------------------------------------------
struct UniformBufferObject
{
    SLMat4f model;
    SLMat4f view;
    SLMat4f proj;
};
//-----------------------------------------------------------------------------
/*
struct Vertex
{
    SLVec2f pos;
    SLVec3f color;
    SLVec3f texCoord;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding                         = 0;
        bindingDescription.stride                          = sizeof(Vertex);
        bindingDescription.inputRate                       = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
    {
        array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

        attributeDescriptions[0].binding  = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format   = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset   = offsetof(Vertex, pos);

        attributeDescriptions[1].binding  = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset   = offsetof(Vertex, color);

        attributeDescriptions[2].binding  = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format   = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset   = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
};
*/
//-----------------------------------------------------------------------------
class vkUtils
{
public:
    // How many frames should be created (after creating image, the device goes into idle mode)
    const int MAX_FRAMES_PROCESSING_ROW = 2;

    const vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                                  VK_KHR_MAINTENANCE1_EXTENSION_NAME};

private:
#pragma region Device
    VkSurfaceKHR     surface;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device;
    VkQueue          graphicsQueue;
    VkQueue          presentQueue;
    VkCommandPool    commandPool;
    vector<VkFence>  inFlightFences;
    vector<VkFence>  imagesInFlight;
#pragma endregion

    VkInstance                      instance;
    VkDebugUtilsMessengerEXT        debugMessenger;
    VkSwapchainKHR                  swapchain;
    vector<VkImage>                 swapchainImages;
    VkFormat                        swapchainImageFormat;
    VkExtent2D                      swapchainExtent;
    vector<VkImageView>             swapchainImageViews;
    vector<VkFramebuffer>           swapchainFramebuffers;
    VkRenderPass                    renderPass;
    VkDescriptorSetLayout           descriptorSetLayout;
    VkDescriptorPool                descriptorPool;
    VkPipelineLayout                pipelineLayout;
    VkPipeline                      graphicsPipeline;
    VkBuffer                        stagingBuffer;
    VkDeviceMemory                  stagingBufferMemory;
    VkImageView                     textureImageView;
    VkSampler                       textureSampler;
    VkBuffer                        indexBuffer;
    VkPipelineShaderStageCreateInfo shaderStages[2];
    vector<VkDescriptorSet>         descriptorSets;
    vector<VkBuffer>                uniformBuffers;
    vector<VkDeviceMemory>          uniformBuffersMemory;
    vector<VkCommandBuffer>         commandBuffers;
    vector<VkSemaphore>             imageAvailableSemaphores;
    vector<VkSemaphore>             renderFinishedSemaphores;
    size_t                          currentFrame       = 0;
    bool                            framebufferResized = false;
    const vector<uint16_t>          indices            = {0, 1, 2, 2, 3, 0};
    SLMat4f*                        cameraMatrix;

public:
    void     drawFrame();
    void     cleanup();
    void     createInstance();
    void     setupDebugMessenger();
    void     createSurface(GLFWwindow*);
    void     pickPhysicalDevice();
    void     createLogicalDevice();
    void     createSwapchain(GLFWwindow*);
    void     createImageViews();
    void     createRenderPass();
    void     createDescriptorSetLayout();
    void     createShaderStages(string& vertShaderPath, string& fragShaderPath);
    void     createGraphicsPipeline();
    void     createFramebuffers();
    void     createCommandPool();
    void     createTextureImage(void* pixels, uint width, uint height);
    void     createTextureSampler();
    VkBuffer createVertexBuffer(const vector<Vertex>& vertices);
    void     createIndexBuffer();
    void     createUniformBuffers();
    void     createDescriptorPool();
    void     createDescriptorSets();
    void     createCommandBuffers(const vector<Vertex>& vertices);
    void     createSyncObjects();
    void     setCameraMatrix(SLMat4f*);
    void     recreateSwapchain(GLFWwindow* window, const vector<Vertex>& vertices);

private:
    void     cleanupSwapchain();
    void     populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT&);
    void     createImage(uint32_t,
                         uint32_t,
                         VkFormat,
                         VkImageTiling,
                         VkImageUsageFlags,
                         VkMemoryPropertyFlags,
                         VkImage&);
    void     transitionImageLayout(VkImage,
                               VkFormat,
                               VkImageLayout,
                               VkImageLayout);
    void     copyBufferToImage(VkBuffer,
                           VkImage,
                           uint32_t,
                           uint32_t);
    void     createBuffer(VkDeviceSize,
                          VkBufferUsageFlags,
                          VkMemoryPropertyFlags,
                          VkBuffer&,
                          VkDeviceMemory&);
    void     copyBuffer(VkBuffer, VkBuffer, VkDeviceSize);
    void     updateUniformBuffer(uint32_t);
    void     DestroyDebugUtilsMessengerEXT(VkInstance,
                                           VkDebugUtilsMessengerEXT,
                                           const VkAllocationCallbacks*);
    bool     checkValidationLayerSupport();
    bool     isDeviceSuitable(VkPhysicalDevice);
    bool     checkDeviceExtensionSupport(VkPhysicalDevice);
    uint32_t findMemoryType(uint32_t,
                            VkMemoryPropertyFlags);

    VkCommandBuffer         beginSingleTimeCommands();
    void                    endSingleTimeCommands(VkCommandBuffer);
    VkImageView             createImageView(VkImage, VkFormat);
    VkShaderModule          createShaderModule(const vector<char>&);
    VkSurfaceFormatKHR      chooseSwapSurfaceFormat(const vector<VkSurfaceFormatKHR>&);
    VkPresentModeKHR        chooseSwapPresentMode(const vector<VkPresentModeKHR>&);
    VkExtent2D              chooseSwapExtent(const VkSurfaceCapabilitiesKHR&, GLFWwindow*);
    SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice);
    QueueFamilyIndices      findQueueFamilies(VkPhysicalDevice);
    vector<const char*>     getRequiredExtensions();

    VkResult CreateDebugUtilsMessengerEXT(VkInstance,
                                          const VkDebugUtilsMessengerCreateInfoEXT*,
                                          const VkAllocationCallbacks*,
                                          VkDebugUtilsMessengerEXT*);

public:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        auto app                = reinterpret_cast<vkUtils*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

private:
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                                        VkDebugUtilsMessageTypeFlagsEXT             messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        void*                                       pUserData)
    {
        cerr << "validation layer: " << pCallbackData->pMessage << endl;
        return VK_FALSE;
    }
};
//-----------------------------------------------------------------------------
static vector<char> readFile(const string& filename)
{
    ifstream file(filename, ios::ate | ios::binary);

    if (!file.is_open())
        throw runtime_error("failed to open file!");

    size_t       fileSize = (size_t)file.tellg();
    vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}
//-----------------------------------------------------------------------------
