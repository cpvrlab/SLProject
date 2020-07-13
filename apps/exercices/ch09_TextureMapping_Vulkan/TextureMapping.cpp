#define GLFW_INCLUDE_VULKAN
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <math/SLVec3.h>
#include <Node.h>
#include <Sphere.h>
#include <Rectangle.h>
#include <VulkanRenderer.h>
#include <vkEnums.h>
#include <DrawingObject.h>

//-----------------------------------------------------------------------------
//////////////////////
// Global Variables //
//////////////////////
const int WINDOW_WIDTH   = 800;
const int WINDOW_HEIGHT  = 600;
string    vertShaderPath = SLstring(SL_PROJECT_ROOT) + "/data/shaders/vertShader.vert.spv";
string    fragShaderPath = SLstring(SL_PROJECT_ROOT) + "/data/shaders/fragShader.frag.spv";

GLFWwindow* window;

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
void createScene(Node& root)
{
    // Mesh 1
    Texture*  texture1  = new Texture("Tree", SLstring(SL_PROJECT_ROOT) + "/data/images/textures/tree1_1024_C.png");
    Material* material1 = new Material("Texture");
    material1->addTexture(texture1);
    GPUProgram* gpuProgram1 = new GPUProgram("First_Shader");
    GPUShader*  vertShader1 = new GPUShader("vertShader", SLstring(SL_PROJECT_ROOT) + "/data/shaders/vertShader.vert.spv", ShaderType::VERTEX);
    GPUShader*  fragShader1 = new GPUShader("fragShader", SLstring(SL_PROJECT_ROOT) + "/data/shaders/fragShader.frag.spv", ShaderType::FRAGMENT);
    gpuProgram1->addShader(vertShader1);
    gpuProgram1->addShader(fragShader1);
    material1->setProgram(gpuProgram1);
    Mesh* mesh1 = new Sphere("Simple_Sphere", 1.0f, 32, 32);
    mesh1->setColor(SLCol4f(1.0f, 1.0f, 1.0f, 1.0f));
    mesh1->mat  = material1;
    Node* node1 = new Node("Sphere");
    node1->om(SLMat4f(0.0f, 0.0f, 0.0f));
    node1->SetMesh(mesh1);

    // Mesh 2
    Texture*  texture2  = new Texture("Tree", SLstring(SL_PROJECT_ROOT) + "/data/images/textures/tree1_1024_C.png");
    Material* material2 = new Material("Texture");
    material2->addTexture(texture2);
    GPUProgram* gpuProgram2 = new GPUProgram("First_Shader");
    GPUShader*  vertShader2 = new GPUShader("vertShaderRed", SLstring(SL_PROJECT_ROOT) + "/data/shaders/vertShaderRed.vert.spv", ShaderType::VERTEX);
    GPUShader*  fragShader2 = new GPUShader("fragShaderRed", SLstring(SL_PROJECT_ROOT) + "/data/shaders/fragShaderRed.frag.spv", ShaderType::FRAGMENT);
    gpuProgram2->addShader(vertShader2);
    gpuProgram2->addShader(fragShader2);
    material2->setProgram(gpuProgram2);
    Mesh* mesh2 = new Sphere("Simple_Sphere", 1.0f, 32, 32);
    mesh2->setColor(SLCol4f(1.0f, 0.0f, 0.0f, 1.0f));
    mesh2->mat  = material2;
    Node* node2 = new Node("Sphere");
    node2->om(SLMat4f(1.0f, 0.0f, 0.0f));
    node2->SetMesh(mesh2);

    root.AddChild(node1);
    root.AddChild(node2);
}
//-----------------------------------------------------------------------------
void SceneToMaterial(Node& root, vector<DrawingObject>& drawingObjectList)
{
    for (Node* node : root.children())
    {
        if (node != nullptr)
            SceneToMaterial(*node, drawingObjectList);

        Material* m     = node->mesh()->mat;
        int       index = INT_MAX;

        for (size_t i = 0; i < drawingObjectList.size(); i++)
            if (drawingObjectList[i].mat == m)
                index = i;

        if (index == INT_MAX)
        {
            DrawingObject drawingObj = DrawingObject();
            drawingObj.mat           = m;
            drawingObjectList.push_back(drawingObj);
            index = drawingObjectList.size() - 1;
        }

        drawingObjectList[index].nodeList.push_back(node);
    }
}
//-----------------------------------------------------------------------------
int main()
{
    initWindow();
    // Create a sphere
    Node root = Node("Root");
    createScene(root);
    vector<DrawingObject> drawingObjectList;
    SceneToMaterial(root, drawingObjectList);

    VulkanRenderer renderer(window);
    renderer.createMesh(_viewMatrix, drawingObjectList);

    // Render
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        updateCamera();
        renderer.draw();
        printFPS();
    }

    return EXIT_SUCCESS;
}
//-----------------------------------------------------------------------------
