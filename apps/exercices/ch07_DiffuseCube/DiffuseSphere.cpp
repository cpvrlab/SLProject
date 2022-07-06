// #############################################################################
//   File:      DiffuseSphere.cpp
//   Purpose:   Core profile OpenGL application with diffuse lighted sphere with
//              GLFW as the OS GUI interface (http://www.glfw.org/).
//   Date:      December 2015 (HS15)
//   Authors:   Marcus Hudritsch
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
// #############################################################################

#include <GL/gl3w.h>    // OpenGL headers
#include <GLFW/glfw3.h> // GLFW GUI library
#include <SLMat4.h>     // 4x4 matrix class
#include <SLVec3.h>     // 3D vector class
#include <glUtils.h>    // Basics for OpenGL shaders, buffers & textures

//-----------------------------------------------------------------------------
//! Struct definition for vertex attributes
struct VertexPN
{
    SLVec3f p; // vertex position [x,y,z]
    SLVec3f n; // vertex normal [x,y,z]

    // Setter method
    void set(float posX, float posY, float posZ, float normalX, float normalY, float normalZ)
    {
        p.set(posX, posY, posZ);
        n.set(normalX, normalY, normalZ);
    }
};
//-----------------------------------------------------------------------------
// Global application variables
GLFWwindow*     window;       //!< The global glfw window handle
static SLstring _projectRoot; //!< Directory of executable
SLint           _scrWidth;    //!< Window width at start up
SLint           _scrHeight;   //!< Window height at start up

SLMat4f _viewMatrix;       //!< 4x4 view matrix
SLMat4f _modelMatrix;      //!< 4x4 model matrix
SLMat4f _projectionMatrix; //!< 4x4 projection matrix

GLuint _vao  = 0; //!< ID of the Vertex Array Object (VAO)
GLuint _vboV = 0; //!< ID of the VBO for vertex array
GLuint _vboI = 0; //!< ID of the VBO for vertex index array

GLuint _numV = 0;      //!< NO. of vertices
GLuint _numI = 0;      //!< NO. of vertex indexes for triangles
GLint  _resolution;    //!< resolution of sphere stack & slices
GLint  _primitiveType; //!< Type of GL primitive to render

float        _camZ;                   //!< z-distance of camera
float        _rotX, _rotY;            //!< rotation angles around x & y axis
int          _deltaX, _deltaY;        //!< delta mouse motion
int          _startX, _startY;        //!< x,y mouse start positions
int          _mouseX, _mouseY;        //!< current mouse position
bool         _mouseLeftDown;          //!< Flag if mouse is down
GLuint       _modifiers = 0;          //!< modifier bit flags
const GLuint NONE       = 0;          //!< constant for no modifier
const GLuint SHIFT      = 0x00200000; //!< constant for shift key modifier
const GLuint CTRL       = 0x00400000; //!< constant for control key modifier
const GLuint ALT        = 0x00800000; //!< constant for alt key modifier

GLuint _shaderVertID = 0; //! vertex shader id
GLuint _shaderFragID = 0; //! fragment shader id
GLuint _shaderProgID = 0; //! shader program id

// Attribute & uniform variable location indexes
GLint _pLoc;            //!< attribute location for vertex position
GLint _nLoc;            //!< attribute location for vertex normal
GLint _pmLoc;           //!< uniform location for projection matrix
GLint _vmLoc;           //!< uniform location for view matrix
GLint _mmLoc;           //!< uniform location for model matrix
GLint _lightDirVSLoc;   //!< uniform location for light direction in VS
GLint _lightDiffuseLoc; //!< uniform location for diffuse light intensity
GLint _matDiffuseLoc;   //!< uniform location for diffuse light reflection

static const SLfloat PI = 3.14159265358979f;

//-----------------------------------------------------------------------------
/*!
buildSphere creates the vertex attributes for a sphere and creates the VBO
at the end. The sphere is built in stacks & slices and the primitive type can
be GL_TRIANGLES or GL_TRIANGLE_STRIP.
*/
void buildSphere(float radius, int stacks, int slices, GLuint primitveType)
{
    assert(stacks > 3 && slices > 3);
    assert(primitveType == GL_TRIANGLES || primitveType == GL_TRIANGLE_STRIP);

    // Create vertex array
    VertexPN* vertices = 0; //!< Array of vertices
    // ???

    // create Index array
    GLuint* indices = 0;
    // ???

    // Generate the OpenGL vertex array object
    if (vertices && indices)
    {
        glUtils::buildVAO(_vao, _vboV, _vboI, vertices, _numV, sizeof(VertexPN), indices, _numI, sizeof(GL_UNSIGNED_INT), _shaderProgID, _pLoc, _nLoc);

        // Delete arrays on heap
        delete[] vertices;
        delete[] indices;
    }
}
//-----------------------------------------------------------------------------
/*!
calcFPS determines the frame per second measurement by averaging 60 frames.
*/
float calcFPS(float deltaTime)
{
    const SLint    FILTERSIZE = 60;
    static SLfloat frameTimes[FILTERSIZE];
    static SLuint  frameNo = 0;

    frameTimes[frameNo % FILTERSIZE] = deltaTime;
    float sumTime                    = 0.0f;
    for (SLuint i = 0; i < FILTERSIZE; ++i) sumTime += frameTimes[i];
    frameNo++;
    float frameTimeSec = sumTime / (SLfloat)FILTERSIZE;
    float fps          = 1 / frameTimeSec;

    return fps;
}
//-----------------------------------------------------------------------------
/*!
onInit initializes the global variables and builds the shader program. It
should be called after a window with a valid OpenGL context is present.
*/
void onInit()
{
    // backwards movement of the camera
    _camZ = -4;

    // Mouse rotation parameters
    _rotX = _rotY = 0;
    _deltaX = _deltaY = 0;
    _mouseLeftDown    = false;

    // Load, compile & link shaders
    _shaderVertID = glUtils::buildShader("../_data/shaders/Diffuse.vert", GL_VERTEX_SHADER);
    _shaderFragID = glUtils::buildShader("../_data/shaders/Diffuse.frag", GL_FRAGMENT_SHADER);
    _shaderProgID = glUtils::buildProgram(_shaderVertID, _shaderFragID);

    // Activate the shader program
    glUseProgram(_shaderProgID);

    // Get the variable locations (identifiers) within the program
    _pLoc            = glGetAttribLocation(_shaderProgID, "a_position");
    _nLoc            = glGetAttribLocation(_shaderProgID, "a_normal");
    _pmLoc           = glGetUniformLocation(_shaderProgID, "u_pMatrix");
    _vmLoc           = glGetUniformLocation(_shaderProgID, "u_vMatrix");
    _mmLoc           = glGetUniformLocation(_shaderProgID, "u_mMatrix");
    _lightDirVSLoc   = glGetUniformLocation(_shaderProgID, "u_lightDirVS");
    _lightDiffuseLoc = glGetUniformLocation(_shaderProgID, "u_lightDiff");
    _matDiffuseLoc   = glGetUniformLocation(_shaderProgID, "u_matDiff");

    // Create sphere
    _resolution    = 16;
    _primitiveType = GL_TRIANGLE_STRIP;
    buildSphere(1.0f, _resolution, _resolution, _primitiveType);

    glClearColor(0.5f, 0.5f, 0.5f, 1); // Set the background color
    glEnable(GL_DEPTH_TEST);           // Enables depth test
    glEnable(GL_CULL_FACE);            // Enables the culling of back faces
}
//-----------------------------------------------------------------------------
/*!
onClose is called when the user closes the window and can be used for proper
deallocation of resources.
*/
void onClose(GLFWwindow* myWindow)
{
    // Delete shaders & programs on GPU
    glDeleteShader(_shaderVertID);
    glDeleteShader(_shaderFragID);
    glDeleteProgram(_shaderProgID);

    // Delete arrays & buffers on GPU
    glDeleteVertexArrays(1, &_vao);
    glDeleteBuffers(1, &_vboV);
    glDeleteBuffers(1, &_vboI);
}
//-----------------------------------------------------------------------------
/*!
onPaint does all the rendering for one frame from scratch with OpenGL.
*/
bool onPaint()
{
    // Clear the color & depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // View transform: move the coordinate system away from the camera
    _viewMatrix.identity();
    _viewMatrix.translate(0, 0, _camZ);

    // Model transform: rotate the coordinate system increasingly
    _viewMatrix.rotate(_rotX + _deltaX, 1, 0, 0);
    _viewMatrix.rotate(_rotY + _deltaY, 0, 1, 0);

    // Model transform: move the cube so that it rotates around its center
    _modelMatrix.identity();

    // Pass the uniform variables to the shader
    glUniformMatrix4fv(_pmLoc, 1, 0, (float*)&_projectionMatrix);
    glUniformMatrix4fv(_vmLoc, 1, 0, (float*)&_viewMatrix);
    glUniformMatrix4fv(_mmLoc, 1, 0, (float*)&_modelMatrix);
    glUniform3f(_lightDirVSLoc, 0.5f, 1.0f, 1.0f);         // light direction in view space
    glUniform4f(_lightDiffuseLoc, 1.0f, 1.0f, 1.0f, 1.0f); // diffuse light intensity (RGBA)
    glUniform4f(_matDiffuseLoc, 1.0f, 0.0f, 0.0f, 1.0f);   // diffuse material reflection (RGBA)

    // Activate
    glBindVertexArray(_vao);

    // Activate index VBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _vboI);

    //////////////////////////////////////////////////////////
    glDrawElements(_primitiveType, _numI, GL_UNSIGNED_INT, 0);
    //////////////////////////////////////////////////////////

    // Deactivate VAO
    glBindVertexArray(_vao);

    // Fast copy the back buffer to the front buffer. This is OS dependent.
    glfwSwapBuffers(window);

    // Calculate frames per second
    char         title[255];
    static float lastTimeSec = 0;
    float        timeNowSec  = (float)glfwGetTime();
    float        fps         = calcFPS(timeNowSec - lastTimeSec);
    string       prim        = _primitiveType == GL_TRIANGLES ? "GL_TRIANGLES" : "GL_TRIANGLE_STRIPS";
    sprintf(title, "Sphere, %d x %d, fps: %4.0f, %s", _resolution, _resolution, fps, prim.c_str());
    glfwSetWindowTitle(window, title);
    lastTimeSec = timeNowSec;

    // Return true to get an immediate refresh
    return true;
}
//-----------------------------------------------------------------------------
/*!
onResize: Event handler called on the resize event of the window. This event
should called once before the onPaint event. Do everything that is dependent on
the size and ratio of the window.
*/
void onResize(GLFWwindow* myWindow, int width, int height)
{
    double w = (double)width;
    double h = (double)height;

    // define the projection matrix
    _projectionMatrix.perspective(50, w / h, 0.01f, 10.0f);

    // define the viewport
    glViewport(0, 0, width, height);

    onPaint();
}
//-----------------------------------------------------------------------------
/*!
Mouse button down & release eventhandler starts and end mouse rotation
*/
void onMouseButton(GLFWwindow* myWindow, int button, int action, int mods)
{
    SLint x = _mouseX;
    SLint y = _mouseY;

    _mouseLeftDown = (action == GLFW_PRESS);
    if (_mouseLeftDown)
    {
        _startX = x;
        _startY = y;

        // Renders only the lines of a polygon during mouse moves
        if (button == GLFW_MOUSE_BUTTON_RIGHT)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    else
    {
        _rotX += _deltaX;
        _rotY += _deltaY;
        _deltaX = 0;
        _deltaY = 0;

        // Renders filled polygons
        if (button == GLFW_MOUSE_BUTTON_RIGHT)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
}
//-----------------------------------------------------------------------------
/*!
Mouse move eventhandler tracks the mouse delta since touch down (_deltaX/_deltaY)
*/
void onMouseMove(GLFWwindow* myWindow, double x, double y)
{
    _mouseX = (int)x;
    _mouseY = (int)y;

    if (_mouseLeftDown)
    {
        _deltaY = (int)x - _startX;
        _deltaX = (int)y - _startY;
        onPaint();
    }
}
//-----------------------------------------------------------------------------
/*!
Mouse wheel eventhandler that moves the camera foreward or backwards
*/
void onMouseWheel(GLFWwindow* myWindow, double xscroll, double yscroll)
{
    if (_modifiers == NONE)
    {
        _camZ += (SLfloat)Utils::sign(yscroll) * 0.1f;
        onPaint();
    }
}
//-----------------------------------------------------------------------------
/*!
Key action eventhandler handles key down & release events
*/
void onKey(GLFWwindow* myWindow, int GLFWKey, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (GLFWKey)
        {
            case GLFW_KEY_ESCAPE:
                onClose(window);
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_RIGHT:
                _resolution    = _resolution << 1;
                _primitiveType = GL_TRIANGLE_STRIP;
                buildSphere(1.0f, _resolution, _resolution, _primitiveType);
                break;
            case GLFW_KEY_LEFT:
                _primitiveType = GL_TRIANGLE_STRIP;
                if (_resolution > 4) _resolution = _resolution >> 1;
                buildSphere(1.0f, _resolution, _resolution, _primitiveType);
                break;
            case GLFW_KEY_UP:
                _resolution    = _resolution << 1;
                _primitiveType = GL_TRIANGLES;
                buildSphere(1.0f, _resolution, _resolution, _primitiveType);
                break;
            case GLFW_KEY_DOWN:
                _primitiveType = GL_TRIANGLES;
                if (_resolution > 4) _resolution = _resolution >> 1;
                buildSphere(1.0f, _resolution, _resolution, _primitiveType);
                break;
        }
    }
}

//-----------------------------------------------------------------------------
/*!
Error callback handler for GLFW.
*/
void onGLFWError(int error, const char* description)
{
    fputs(description, stderr);
}

//-----------------------------------------------------------------------------
/*!
The C main procedure running the GLFW GUI application.
*/
int main(int argc, char* argv[])
{
    _projectRoot = SLstring(SL_PROJECT_ROOT);

    // Initialize the platform independent GUI-Library GLFW
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(onGLFWError);

    // Enable fullscreen anti aliasing with 4 samples
    glfwWindowHint(GLFW_SAMPLES, 4);

    // You can enable or restrict newer OpenGL context here (read the GLFW documentation)
#ifdef __APPLE__
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GL_FALSE);
#else
    // glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    _scrWidth  = 640;
    _scrHeight = 480;

    // Create the GLFW window
    window = glfwCreateWindow(_scrWidth,
                              _scrHeight,
                              "Diffuse Sphere",
                              nullptr,
                              nullptr);

    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Get the current GL context. After this you can call GL
    glfwMakeContextCurrent(window);

    // Init OpenGL access library gl3w
    if (gl3wInit() != 0)
    {
        std::cerr << "Failed to initialize OpenGL" << std::endl;
        exit(-1);
    }

    // Check errors before we start
    GETGLERROR;

    glUtils::printGLInfo();

    // Set number of monitor refreshes between 2 buffer swaps
    glfwSwapInterval(1);

    // Prepare all OpenGL stuff
    onInit();

    // Call resize once for correct projection
    onResize(window, _scrWidth, _scrHeight);

    // Set GLFW callback functions
    glfwSetKeyCallback(window, onKey);
    glfwSetFramebufferSizeCallback(window, onResize);
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetCursorPosCallback(window, onMouseMove);
    glfwSetScrollCallback(window, onMouseWheel);
    glfwSetWindowCloseCallback(window, onClose);

    // Event loop
    while (!glfwWindowShouldClose(window))
    {
        // if no updated occurred wait for the next event (power saving)
        if (!onPaint())
            glfwWaitEvents();
        else
            glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(0);
}
//-----------------------------------------------------------------------------
