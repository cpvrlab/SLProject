#include "vkUtils.h"

#include <GLFW/glfw3.h>
#include <string>

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
std::string vertShaderPath = SLstring(SL_PROJECT_ROOT) + "/data/shaders/vertShader.spv";
std::string fragShaderPath = SLstring(SL_PROJECT_ROOT) + "/data/shaders/fragShader.spv";

GLFWwindow* window;
vkUtils     renderer;


void initWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, &renderer);
    glfwSetFramebufferSizeCallback(window, renderer.framebufferResizeCallback);
}

void initVulkan()
{
    CVImage texture;
    texture.load(SLstring(SL_PROJECT_ROOT) + "/data/images/textures/tree1_1024_C.png", false, false);

    renderer.createInstance(window);
    renderer.setupDebugMessenger();
    renderer.createSurface();
    renderer.pickPhysicalDevice();
    renderer.createLogicalDevice();
    renderer.createSwapchain();
    renderer.createImageViews();
    renderer.createRenderPass();
    renderer.createDescriptorSetLayout();
    renderer.createShaderStages(vertShaderPath, fragShaderPath);
    renderer.createGraphicsPipeline();
    renderer.createFramebuffers();
    renderer.createCommandPool();
    renderer.createTextureImage(texture.data(), texture.width(), texture.height());
    renderer.createTextureImageView();
    renderer.createTextureSampler();
    renderer.createVertexBuffer();
    renderer.createIndexBuffer();
    renderer.createUniformBuffers();
    renderer.createDescriptorPool();
    renderer.createDescriptorSets();
    renderer.createCommandBuffers();
    renderer.createSyncObjects();

    texture.~CVImage();
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        renderer.drawFrame();
    }

    renderer.deviceWaitIdle();
}

void cleanup()
{
    renderer.cleanup();

    glfwDestroyWindow(window);
    glfwTerminate();
}

int main()
{
    try
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}