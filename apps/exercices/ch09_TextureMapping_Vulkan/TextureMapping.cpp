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
#include <Camera.h>
#include <random>

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
// SLMat4f _viewMatrix;
float  _camZ                  = 6.0f;
float  _mouseWheelSensitivity = 0.5f;
Camera camera;

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
    _camZ -= (SLfloat)Utils::sign(yScroll) * _mouseWheelSensitivity;
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
    // static float updateTimeSec = 0.0f;
    float timeNowSec = (float)glfwGetTime();
    float fps        = calcFPS(timeNowSec - lastTimeSec);
    // if ((timeNowSec - updateTimeSec) >= 0.001f)
    {
        sprintf(title, "fps: %5.0f", fps);
        glfwSetWindowTitle(window, title);
        //  updateTimeSec = timeNowSec;
    }
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
    camera.om().identity();
    camera.om().translate(0.0f, 0.0f, -_camZ);
    camera.om().rotate((float)(_rotX + _deltaX), 1.0f, 0.0f, 0.0f);
    camera.om().rotate((float)(_rotY + _deltaY), 0.0f, 1.0f, 0.0f);
}
//-----------------------------------------------------------------------------
void createScene(Node& root)
{
    int   sizeX           = 1; // 17;
    int   sizeY           = 1; // 17;
    int   sizeZ           = 1; // 17;
    int   materialCount   = 1; // 7 * 7;
    float offsetDimension = 2.5f;

    float offsetX = (sizeX % 2 != 0) ? 0.0f : 0.5f;
    float offsetY = (sizeY % 2 != 0) ? 0.0f : 0.5f;
    float offsetZ = (sizeZ % 2 != 0) ? 0.0f : 0.5f;

    vector<Material*> materialList = vector<Material*>(materialCount);

    for (int x = 0; x < materialCount; x++)
    {
        Texture*  texture  = new Texture("Tree", SLstring(SL_PROJECT_ROOT) + "/data/images/textures/earth1024_C_alpha.png");
        Material* material = new Material("Texture");
        material->addTexture(texture);
        GPUProgram* gpuProgram = new GPUProgram("First_Shader");
        GPUShader*  vertShader = new GPUShader("vertShader", SLstring(SL_PROJECT_ROOT) + "/data/shaders/vertShader.vert.spv", ShaderType::VERTEX);
        GPUShader*  fragShader = new GPUShader("fragShader", SLstring(SL_PROJECT_ROOT) + "/data/shaders/fragShader.frag.spv", ShaderType::FRAGMENT);
        gpuProgram->addShader(vertShader);
        gpuProgram->addShader(fragShader);
        material->setProgram(gpuProgram);

        materialList[x] = material;
    }

    for (int x = 0; x < sizeX; x++)
    {
        for (int y = 0; y < sizeY; y++)
        {
            for (int z = 0; z < sizeZ; z++)
            {

                Mesh* mesh = new Sphere("Simple_Sphere", 1.0f, 18, 18);

                float rR = random(0.0f, 1.0f);
                float rG = random(0.0f, 1.0f);
                float rB = random(0.0f, 1.0f);
                mesh->setColor(SLCol4f(rR, rG, rB, 1.0f));

                int randomMaterialIndex = (int)random(0, materialCount);
                mesh->mat               = materialList[randomMaterialIndex];
                Node* node              = new Node("Sphere");
                node->om(SLMat4f((x + offsetX - (sizeX / 2)) * offsetDimension,
                                 (y + offsetY - (sizeY / 2)) * offsetDimension,
                                 (z + offsetZ - (sizeZ / 2)) * offsetDimension));
                node->SetMesh(mesh);

                root.AddChild(node);
            }
        }
    }

#if 0
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
    node1->om(SLMat4f(-2.0f, 0.0f, 0.0f));
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
    node2->om(SLMat4f(2.0f, 0.0f, 0.0f));
    node2->SetMesh(mesh2);

    root.AddChild(node1);
    root.AddChild(node2);
#endif
}
//-----------------------------------------------------------------------------
void SceneToMaterialCompromissed(Node& root, vector<DrawingObject>& objectsInScene)
{
    for (Node* node : root.children())
    {
        if (node != nullptr)
            SceneToMaterialCompromissed(*node, objectsInScene);

        Material* m     = node->mesh()->mat;
        int       index = INT_MAX;

        for (size_t i = 0; i < objectsInScene.size(); i++)
            if (objectsInScene[i].mat == m)
                index = i;

        if (index == INT_MAX)
        {
            DrawingObject drawingObj = DrawingObject();
            drawingObj.mat           = m;
            objectsInScene.push_back(drawingObj);
            index = objectsInScene.size() - 1;
        }

        objectsInScene[index].nodeList.push_back(node);
    }
}
//-----------------------------------------------------------------------------
void SceneToMaterialSimple(Node& root, vector<DrawingObject>& objectsInScene)
{
    for (Node* node : root.children())
    {
        if (node != nullptr)
            SceneToMaterialCompromissed(*node, objectsInScene);

        DrawingObject drawingObj = DrawingObject();
        drawingObj.mat           = node->mesh()->mat;
        objectsInScene.push_back(drawingObj);
        int index = objectsInScene.size() - 1;
        objectsInScene[index].nodeList.push_back(node);
    }
}
//-----------------------------------------------------------------------------
int main()
{
    initWindow();
    camera.setViewport(WINDOW_WIDTH, WINDOW_HEIGHT);
    // Create a sphere
    Node root = Node("Root");
    createScene(root);
    vector<DrawingObject> objectsInScene;
    SceneToMaterialCompromissed(root, objectsInScene);

    VulkanRenderer renderer(window);
    renderer.createMesh(camera, objectsInScene);

    // TODO:
    // - Create other vector<DrawingObject> for actural drawing in scene
    // - Loop checking each nodes frustum and take it from objectsInScene
    // - Clear list after each frame (or check if camera has been moved)

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
