#include "vkUtils.h"

#include <GLFW/glfw3.h>
#include <string>
#include <CVImage.h> // Image class for image loading

//-----------------------------------------------------------------------------
//////////////////////
// Global Variables //
//////////////////////

const int WINDOW_WIDTH   = 800;
const int WINDOW_HEIGHT  = 600;
string    vertShaderPath = SLstring(SL_PROJECT_ROOT) + "/data/shaders/vertShader.spv";
string    fragShaderPath = SLstring(SL_PROJECT_ROOT) + "/data/shaders/fragShader.spv";

GLFWwindow*          window;
vkUtils              renderer;
const vector<Vertex> vertices = {{{-1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
                                 {{1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                                 {{1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                                 {{-1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}}};
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
    _mouseLeftDown = (action == GLFW_PRESS);
    if (_mouseLeftDown)
    {
        _startX = _mouseX;
        _startY = _mouseY;
    }
    else
    {
        _rotX -= _deltaX;
        _rotY -= _deltaY;
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
        _deltaY = _mouseX - _startX;
        _deltaX = _mouseY - _startY;
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
void onWindowResize(GLFWwindow* window, int width, int height)
{
    renderer.recreateSwapchain(window, vertices);
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
    glfwSetWindowUserPointer(window, &renderer);
    glfwSetFramebufferSizeCallback(window, renderer.framebufferResizeCallback);
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetCursorPosCallback(window, onMouseMove);
    glfwSetScrollCallback(window, onMouseWheel);
    glfwSetWindowSizeCallback(window, onWindowResize);
}
//-----------------------------------------------------------------------------
void initVulkan()
{
    CVImage texture;
    texture.load(SLstring(SL_PROJECT_ROOT) + "/data/images/textures/tree1_1024_C.png",
                 false,
                 false);

    renderer.setCameraMatrix(&_viewMatrix);

    renderer.createInstance();
    renderer.setupDebugMessenger();
    renderer.createSurface(window);
    renderer.pickPhysicalDevice();
    renderer.createLogicalDevice();
    renderer.createSwapchain(window);
    renderer.createImageViews();
    renderer.createRenderPass();
    renderer.createDescriptorSetLayout();
    renderer.createShaderStages(vertShaderPath, fragShaderPath);
    renderer.createGraphicsPipeline();
    renderer.createFramebuffers();
    renderer.createCommandPool();
    renderer.createTextureImage(texture.data(), texture.width(), texture.height());
    renderer.createTextureSampler();
    // VkBuffer vertexBuffer = renderer.createVertexBuffer(vertices);
    renderer.createIndexBuffer();
    renderer.createUniformBuffers();
    renderer.createDescriptorPool();
    renderer.createDescriptorSets();
    renderer.createCommandBuffers(vertices);
    renderer.createSyncObjects();
}
//-----------------------------------------------------------------------------
void updateCamera()
{
    _viewMatrix = SLMat4f();
    _viewMatrix.rotate(_rotX - _deltaX, 1.0f, 0.0f, 0.0f);
    _viewMatrix.rotate(_rotY - _deltaY, 0.0f, 1.0f, 0.0f);
    SLVec3f a = (_viewMatrix.axisZ()) * _camZ;
    _viewMatrix.lookAt(a, SLVec3f(0.0f, 0.0f, 0.0f), SLVec3f(0.0f, 1.0f, 0.0f));
}
//-----------------------------------------------------------------------------
void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        updateCamera();
        renderer.drawFrame();
        printFPS();
    }
}
//-----------------------------------------------------------------------------
void cleanup()
{
    renderer.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
}
//-----------------------------------------------------------------------------
int main()
{
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();

    return EXIT_SUCCESS;
}
//-----------------------------------------------------------------------------
