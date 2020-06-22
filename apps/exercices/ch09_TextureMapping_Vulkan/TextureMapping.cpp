#define oldProject 0

#define GLFW_INCLUDE_VULKAN
// #include <GL/glew.h> // OpenGL headers
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <string>
#include <CVImage.h> // Image class for image loading
#include <math/SLVec3.h>
#include <glUtils.h>
#include <Utils.h>

#include "Instance.h"
#include "Device.h"
#include "Swapchain.h"
#include "RenderPass.h"
#include "DescriptorSetLayout.h"
#include "ShaderModule.h"
#include "Pipeline.h"
#include "Framebuffer.h"
#include "TextureImage.h"
#include "Sampler.h"
#include "IndexBuffer.h"
#include "UniformBuffer.h"
#include "DescriptorPool.h"
#include "DescriptorSet.h"
#include "VertexBuffer.h"
#include <Node.h>
#include <Sphere.h>

//-----------------------------------------------------------------------------
//////////////////////
// Global Variables //
//////////////////////

struct Vertex;

const int WINDOW_WIDTH   = 800;
const int WINDOW_HEIGHT  = 600;
string    vertShaderPath = SLstring(SL_PROJECT_ROOT) + "/data/shaders/vertShader.vert.spv";
string    fragShaderPath = SLstring(SL_PROJECT_ROOT) + "/data/shaders/fragShader.frag.spv";

GLFWwindow* window;
#if oldProject
vkUtils renderer;
#endif
// Plane
// const vector<Vertex> vertices = {{{-1.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
//                                  {{1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
//                                  {{1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
//                                  {{-1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}}};
// const vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

// Camera
SLMat4f _viewMatrix;
float   _camZ = 6.0f;

// Mouse
int  _startX, _startY;
int  _mouseX, _mouseY;
int  _deltaX, _deltaY;
int  _rotX, _rotY;
bool _mouseLeftDown;

//-----------------------------------------------------------------------------
void onMouseButton(GLFWwindow* window, int button, int action, int mods)
{
    SLint x = _mouseX;
    SLint y = _mouseY;

    _mouseLeftDown = (action == GLFW_PRESS);
    if (_mouseLeftDown)
    {
        _startX = x;
        _startY = y;
    }
    else
    {
        _rotX += _deltaX;
        _rotY += _deltaY;
        _deltaX = 0;
        _deltaY = 0;
    }
}
//-----------------------------------------------------------------------------
void onMouseMove(GLFWwindow* window, double x, double y)
{
    _mouseX = (int)x;
    _mouseY = (int)y;

    if (_mouseLeftDown)
    {
        _deltaY = (int)x - _startX;
        _deltaX = (int)y - _startY;
    }
}
//-----------------------------------------------------------------------------
void onMouseWheel(GLFWwindow* window, double xScroll, double yScroll)
{
    _camZ -= (SLfloat)Utils::sign(yScroll) * 0.1f;
}
//-----------------------------------------------------------------------------
float calcFPS(float deltaTime)
{
    const SLint    FILTERSIZE = 60;
    static SLfloat frameTimes[FILTERSIZE];
    static SLuint  frameNo = 0;

    frameTimes[frameNo % FILTERSIZE] = deltaTime;
    float sumTime                    = 0.0f;

    for (SLuint i = 0; i < FILTERSIZE; ++i)
        sumTime += frameTimes[i];

    frameNo++;
    float frameTimeSec = sumTime / (SLfloat)FILTERSIZE;
    float fps          = 1 / frameTimeSec;

    return fps;
}
//-----------------------------------------------------------------------------
void printFPS()
{
    char         title[255];
    static float lastTimeSec = 0.0f;
    float        timeNowSec  = (float)glfwGetTime();
    float        fps         = calcFPS(timeNowSec - lastTimeSec);
    sprintf(title, "fps: %5.0f", fps);
    glfwSetWindowTitle(window, title);
    lastTimeSec = timeNowSec;
}
//-----------------------------------------------------------------------------
void initWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WINDOW_WIDTH,
                              WINDOW_HEIGHT,
                              "Vulkan",
                              nullptr,
                              nullptr);

    glfwSetWindowSizeLimits(window, 100, 100, GLFW_DONT_CARE, GLFW_DONT_CARE);
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetCursorPosCallback(window, onMouseMove);
    glfwSetScrollCallback(window, onMouseWheel);
}
//-----------------------------------------------------------------------------
void updateCamera()
{
    _viewMatrix.identity();
    _viewMatrix.translate(0.0f, 0.0f, -_camZ);
    _viewMatrix.rotate((float)(_rotX + _deltaX), 1.0f, 0.0f, 0.0f);
    _viewMatrix.rotate((float)(_rotY + _deltaY), 0.0f, 1.0f, 0.0f);
}
//-----------------------------------------------------------------------------
int main()
{
    initWindow();
    // Needed data
    Texture  texture  = Texture("Tree", SLstring(SL_PROJECT_ROOT) + "/data/images/textures/tree1_1024_C.png");
    Material material = Material("Texture");
    material.addTexture(&texture);
    Sphere sphere = Sphere("Simple_Sphere", 0.5f, 32, 32);
    Node   node   = Node("Sphere");
    node.AddMesh(&sphere);

    const vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const vector<const char*> deviceExtensions = {"VK_KHR_swapchain", "VK_KHR_maintenance1"};
    // Setting up vulkan
    Instance     instance = Instance("Test", deviceExtensions, validationLayers);
    VkSurfaceKHR surface;
    glfwCreateWindowSurface(instance.handle, window, nullptr, &surface);
    Device     device     = Device(instance, instance.physicalDevice, surface, deviceExtensions);
    Swapchain  swapchain  = Swapchain(device, window);
    RenderPass renderPass = RenderPass(device, swapchain);

    // Shader program setup
    DescriptorSetLayout descriptorSetLayout = DescriptorSetLayout(device);
    ShaderModule        vertShaderModule    = ShaderModule(device, vertShaderPath);
    ShaderModule        fragShaderModule    = ShaderModule(device, fragShaderPath);
    Pipeline            pipeline            = Pipeline(device, swapchain, descriptorSetLayout, renderPass, vertShaderModule, fragShaderModule);
    Framebuffer         framebuffer         = Framebuffer(device, renderPass, swapchain);

    // Texture setup
    TextureImage textureImage = TextureImage(device, texture.imageData(), texture.imageWidth(), texture.imageHeight());

    // Mesh setup
    Buffer indexBuffer = Buffer(device);
    indexBuffer.createIndexBuffer(sphere.I32);
    UniformBuffer  uniformBuffer  = UniformBuffer(device, swapchain, _viewMatrix, SLMat4f(0.0f, 0.0f, 0.0f));
    DescriptorPool descriptorPool = DescriptorPool(device, swapchain);
    Sampler&       s              = textureImage.sampler();
    DescriptorSet  descriptorSet  = DescriptorSet(device, swapchain, descriptorSetLayout, descriptorPool, uniformBuffer, textureImage.sampler(), textureImage);
    Buffer         vertexBuffer   = Buffer(device);
    vertexBuffer.createVertexBuffer(sphere.P, sphere.N, sphere.Tc, sphere.C, sphere.P.size() / 3);
    // Draw call setup
    CommandBuffer commandBuffer = CommandBuffer(device);
    commandBuffer.setVertices(swapchain, framebuffer, renderPass, vertexBuffer, indexBuffer, pipeline, descriptorSet, (int)sphere.I32.size());
    device.createSyncObjects(swapchain);

    // Render
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        updateCamera();
        pipeline.draw(swapchain, uniformBuffer, commandBuffer);
        printFPS();
    }
    // Destroy
    device.waitIdle();

    framebuffer.destroy();
    commandBuffer.destroy();
    pipeline.destroy();
    renderPass.destroy();
    swapchain.destroy();
    uniformBuffer.destroy();
    descriptorPool.destroy();
    descriptorSetLayout.destroy();
    textureImage.destroy();
    // textureSampler.destroy();
    indexBuffer.destroy();
    vertexBuffer.destroy();
    vertShaderModule.destroy();
    fragShaderModule.destroy();
    device.destroy();
    instance.destroy();

    return EXIT_SUCCESS;
}
//-----------------------------------------------------------------------------
