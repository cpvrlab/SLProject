#include "vkUtils.h"

#include <GLFW/glfw3.h>
#include <string>
#include <CVImage.h> // Image class for image loading

const int   WINDOW_WIDTH   = 800;
const int   WINDOW_HEIGHT  = 600;
std::string vertShaderPath = SLstring(SL_PROJECT_ROOT) + "/data/shaders/vertShader.spv";
std::string fragShaderPath = SLstring(SL_PROJECT_ROOT) + "/data/shaders/fragShader.spv";

GLFWwindow*               window;
vkUtils                   renderer;
const std::vector<Vertex> vertices = {{{-1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
                                      {{1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                                      {{1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                                      {{-1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}}};

// Camera
SLMat4f _viewMatrix;
float _camZ = 6.0f;

// Mouse
int _startX, _startY;
int _mouseX, _mouseY;
int  _deltaX, _deltaY;
int  _rotX, _rotY;
bool _mouseLeftDown;

void onMouseButton(GLFWwindow* window, int button, int action, int mods)
{
    // std::cout << button << " was pressed " << std::endl;

    _mouseLeftDown = (action == GLFW_PRESS);
    if (_mouseLeftDown)
    {
        _startX = _mouseX;
        _startY = _mouseY;
    }
    else
    {
        _rotX += _deltaX;
        _rotY += _deltaY;
        _deltaX = 0;
        _deltaY = 0;
    }
}

void onMouseMove(GLFWwindow* window, double x, double y)
{
    // std::cout << "(" << x << ", " << y << ")" << std::endl;

    _mouseX = (int)x;
    _mouseY = (int)y;

    if (_mouseLeftDown)
    {
        _deltaY = _mouseX - _startX;
        _deltaX = _mouseY - _startY;
    }
}

void initWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, &renderer);
    glfwSetFramebufferSizeCallback(window, renderer.framebufferResizeCallback);
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetCursorPosCallback(window, onMouseMove);
}

void initVulkan()
{
    CVImage texture;
    texture.load(SLstring(SL_PROJECT_ROOT) + "/data/images/textures/tree1_1024_C.png", false, false);

    renderer.setCameraMatrix(&_viewMatrix);

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
    renderer.createVertexBuffer(vertices);
    renderer.createIndexBuffer();
    renderer.createUniformBuffers();
    renderer.createDescriptorPool();
    renderer.createDescriptorSets();
    renderer.createCommandBuffers();
    renderer.createSyncObjects();
}

void updateCamera()
{
    _viewMatrix = SLMat4f();
    _viewMatrix.rotate(_rotX - _deltaX, 1.0f, 0.0f, 0.0f);
    _viewMatrix.rotate(_rotY - _deltaY, 0.0f, 1.0f, 0.0f);
    SLVec3f a = (_viewMatrix.axisZ()) * _camZ;
    _viewMatrix.lookAt(a);
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        updateCamera();
        renderer.drawFrame();
    }
}

void cleanup()
{
    renderer.cleanup();

    glfwDestroyWindow(window);
    glfwTerminate();
}

int main()
{
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();

    return EXIT_SUCCESS;
}