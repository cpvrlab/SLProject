//#############################################################################
//  File:      DiffuseSphere.cpp
//  Purpose:   Core profile OpenGL application with diffuse lighted sphere with
//             GLFW as the OS GUI interface (http://www.glfw.org/).
//  Date:      December 2015 (HS15)
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <GL/gl3w.h>    // OpenGL headers
#include <GLFW/glfw3.h> // GLFW GUI library
#include <SLMat4.h>     // 4x4 matrix class
#include <SLVec3.h>     // 3D vector class
#include <glUtils.h>    // Basics for OpenGL shaders, buffers & textures

//-----------------------------------------------------------------------------
//! Struct definition for vertex attributes position and normal
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
GLFWwindow*     window;                      //!< The global glfw window handle
static SLstring _projectRoot;                //!< Directory of executable
SLint           _scrWidth;                   //!< Window width at start up
SLint           _scrHeight;                  //!< Window height at start up

static SLMat4f _cameraMatrix;                //!< 4x4 matrix for camera to world transform
static SLMat4f _viewMatrix;                  //!< 4x4 matrix for world to camera transform
static SLMat4f _modelMatrix;                 //!< 4x4 matrix for model to world transform
static SLMat4f _projectionMatrix;            //!< Projection from view space to normalized device coordinates

static GLuint _vao  = 0;                     //!< ID of the Vertex Array Object (VAO)
static GLuint _vboV = 0;                     //!< ID of the VBO for vertex array
static GLuint _vboI = 0;                     //!< ID of the VBO for vertex index array

static GLuint _numV = 0;                     //!< NO. of vertices
static GLuint _numI = 0;                     //!< NO. of vertex indexes for triangles
static GLint  _resolution;                   //!< resolution of sphere stack & slices
static GLint  _primitiveType;                //!< Type of GL primitive to render

static float        _camZ;                   //!< z-distance of camera
static float        _rotX, _rotY;            //!< rotation angles around x & y axis
static int          _deltaX, _deltaY;        //!< delta mouse motion
static int          _startX, _startY;        //!< x,y mouse start positions
static int          _mouseX, _mouseY;        //!< current mouse position
static bool         _mouseLeftDown;          //!< Flag if mouse is down
static GLuint       _modifiers = 0;          //!< modifier bit flags
static const GLuint NONE       = 0;          //!< constant for no modifier
static const GLuint SHIFT      = 0x00200000; //!< constant for shift key modifier
static const GLuint CTRL       = 0x00400000; //!< constant for control key modifier
static const GLuint ALT        = 0x00800000; //!< constant for alt key modifier

static GLuint _shaderVertID = 0;             //! vertex shader id
static GLuint _shaderFragID = 0;             //! fragment shader id
static GLuint _shaderProgID = 0;             //! shader program id

// Attribute & uniform variable location indexes
static GLint _pLoc;              //!< attribute location for vertex position
static GLint _nLoc;              //!< attribute location for vertex normal
static GLint _pmLoc;             //!< uniform location for projection matrix
static GLint _vmLoc;             //!< uniform location for view matrix
static GLint _mmLoc;             //!< uniform location for model matrix
static GLint _lightSpotDirVSLoc; //!< uniform location for light direction in view space (VS)
static GLint _lightDiffuseLoc;   //!< uniform location for diffuse light intensity
static GLint _matDiffuseLoc;     //!< uniform location for diffuse light reflection

static const SLfloat PI = 3.14159265358979f;

//-----------------------------------------------------------------------------
/*!
 * Build the vertex and index data for a box and sends it to the GPU
 */
void buildBox()
{
    // create C arrays on heap
    // Define the vertex pos. and normals as an array of structure
    _numV              = 24;
    VertexPN* vertices = new VertexPN[_numV];
    vertices[0].set(1, 1, 1, 1, 0, 0);
    vertices[1].set(1, 0, 1, 1, 0, 0);
    vertices[2].set(1, 0, 0, 1, 0, 0);
    vertices[3].set(1, 1, 0, 1, 0, 0);
    vertices[4].set(1, 1, 0, 0, 0, -1);
    vertices[5].set(1, 0, 0, 0, 0, -1);
    vertices[6].set(0, 0, 0, 0, 0, -1);
    vertices[7].set(0, 1, 0, 0, 0, -1);
    vertices[8].set(0, 0, 1, -1, 0, 0);
    vertices[9].set(0, 1, 1, -1, 0, 0);
    vertices[10].set(0, 1, 0, -1, 0, 0);
    vertices[11].set(0, 0, 0, -1, 0, 0);
    vertices[12].set(1, 1, 1, 0, 0, 1);
    vertices[13].set(0, 1, 1, 0, 0, 1);
    vertices[14].set(0, 0, 1, 0, 0, 1);
    vertices[15].set(1, 0, 1, 0, 0, 1);
    vertices[16].set(1, 1, 1, 0, 1, 0);
    vertices[17].set(1, 1, 0, 0, 1, 0);
    vertices[18].set(0, 1, 0, 0, 1, 0);
    vertices[19].set(0, 1, 1, 0, 1, 0);
    vertices[20].set(0, 0, 0, 0, -1, 0);
    vertices[21].set(1, 0, 0, 0, -1, 0);
    vertices[22].set(1, 0, 1, 0, -1, 0);
    vertices[23].set(0, 0, 1, 0, -1, 0);

    // Define the triangle indexes of the cubes vertices
    _numI           = 36;
    GLuint* indices = new GLuint[_numI];
    int     n       = 0;
    indices[n++]    = 0;
    indices[n++]    = 1;
    indices[n++]    = 2;
    indices[n++]    = 0;
    indices[n++]    = 2;
    indices[n++]    = 3;
    indices[n++]    = 4;
    indices[n++]    = 5;
    indices[n++]    = 6;
    indices[n++]    = 4;
    indices[n++]    = 6;
    indices[n++]    = 7;
    indices[n++]    = 8;
    indices[n++]    = 9;
    indices[n++]    = 10;
    indices[n++]    = 8;
    indices[n++]    = 10;
    indices[n++]    = 11;
    indices[n++]    = 12;
    indices[n++]    = 13;
    indices[n++]    = 14;
    indices[n++]    = 12;
    indices[n++]    = 14;
    indices[n++]    = 15;
    indices[n++]    = 16;
    indices[n++]    = 17;
    indices[n++]    = 18;
    indices[n++]    = 16;
    indices[n++]    = 18;
    indices[n++]    = 19;
    indices[n++]    = 20;
    indices[n++]    = 21;
    indices[n++]    = 22;
    indices[n++]    = 20;
    indices[n++]    = 22;
    indices[n++]    = 23;
    _primitiveType  = GL_TRIANGLES;

    // Generate the OpenGL vertex array object
    glUtils::buildVAO(_vao,
                      _vboV,
                      _vboI,
                      vertices,
                      (GLint)_numV,
                      sizeof(VertexPN),
                      indices,
                      (GLint)_numI,
                      sizeof(GL_UNSIGNED_INT),
                      (GLint)_shaderProgID,
                      _pLoc,
                      -1,
                      _nLoc);

    // Delete arrays on heap. The data for rendering is now on the GPU
    delete[] vertices;
    delete[] indices;
}
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

    // Spherical to cartesian coordinates
    // dtheta = PI  / stacks;
    // dphi = 2 * PI / slices;
    // x = r*sin(theta)*cos(phi);
    // y = r*sin(theta)*sin(phi);
    // z = r*cos(theta);

    // Create vertex array
    VertexPN* vertices = 0; //!< Array of vertices
    // ???

    // create Index array
    GLuint* indices = 0;
    // ???

    // Delete arrays on heap. The data for rendering is now on the GPU
    if (vertices && indices)
    {
        glUtils::buildVAO(_vao, _vboV, _vboI, vertices, _numV, sizeof(VertexPN), indices, _numI, sizeof(GL_UNSIGNED_INT), _shaderProgID, _pLoc, _nLoc);

        // Delete arrays on heap
        delete[] vertices;
        delete[] indices;
    }
    else
        std::cout << "**** You have to define some vertices and indices first in buildSphere! ****" << std::endl;
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
    _camZ = 3;

    // Mouse rotation parameters
    _rotX = _rotY = 0;
    _deltaX = _deltaY = 0;
    _mouseLeftDown    = false;

    // Load, compile & link shaders
    _shaderVertID = glUtils::buildShader(_projectRoot + "/data/shaders/ch07_DiffuseLighting.vert", GL_VERTEX_SHADER);
    _shaderFragID = glUtils::buildShader(_projectRoot + "/data/shaders/ch07_DiffuseLighting.frag", GL_FRAGMENT_SHADER);
    _shaderProgID = glUtils::buildProgram(_shaderVertID, _shaderFragID);

    // Activate the shader program
    glUseProgram(_shaderProgID);

    // Get the variable locations (identifiers) within the program
    _pLoc              = glGetAttribLocation(_shaderProgID, "a_position");
    _nLoc              = glGetAttribLocation(_shaderProgID, "a_normal");
    _pmLoc             = glGetUniformLocation(_shaderProgID, "u_pMatrix");
    _vmLoc             = glGetUniformLocation(_shaderProgID, "u_vMatrix");
    _mmLoc             = glGetUniformLocation(_shaderProgID, "u_mMatrix");
    _lightSpotDirVSLoc = glGetUniformLocation(_shaderProgID, "u_lightSpotDir");
    _lightDiffuseLoc   = glGetUniformLocation(_shaderProgID, "u_lightDiff");
    _matDiffuseLoc     = glGetUniformLocation(_shaderProgID, "u_matDiff");

    // Create sphere
    _resolution    = 16;
    _primitiveType = GL_TRIANGLE_STRIP;

    // buildBox();
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

    /* 2a) Camera transform: rotate the coordinate system increasingly
     * first around the y- and then around the x-axis. This type of camera
     * transform is called turntable animation.*/
    _cameraMatrix.identity();
    _cameraMatrix.rotate(_rotY + _deltaY, 0, 1, 0);
    _cameraMatrix.rotate(_rotX + _deltaX, 1, 0, 0);

    // 2c) Move the camera to its position.
    _cameraMatrix.translate(0, 0, _camZ);

    // 2c) View transform is world to camera (= inverse of camera matrix)
    _viewMatrix = _cameraMatrix.inverted();

    // 3) Model transform: move the cube so that it rotates around its center
    _modelMatrix.identity();

    // 4) Lights get prepared here later on

    // 5) Pass the uniform variables to the shader
    glUniformMatrix4fv(_pmLoc, 1, 0, (float*)&_projectionMatrix);
    glUniformMatrix4fv(_vmLoc, 1, 0, (float*)&_viewMatrix);
    glUniformMatrix4fv(_mmLoc, 1, 0, (float*)&_modelMatrix);
    glUniform3f(_lightSpotDirVSLoc, 0.5f, 1.0f, 1.0f);     // light direction in view space
    glUniform4f(_lightDiffuseLoc, 1.0f, 1.0f, 1.0f, 1.0f); // diffuse light intensity (RGBA)
    glUniform4f(_matDiffuseLoc, 1.0f, 0.0f, 0.0f, 1.0f);   // diffuse material reflection (RGBA)

    // 6) Activate the vertex array
    glBindVertexArray(_vao);

    // 7) Final draw call that draws the cube with triangles by indexes
    glDrawElements(_primitiveType, _numI, GL_UNSIGNED_INT, 0);

    // 8) Fast copy the back buffer to the front buffer. This is OS dependent.
    glfwSwapBuffers(window);

    // Calculate frames per second
    char         title[255];
    static float lastTimeSec = 0;
    float        timeNowSec  = (float)glfwGetTime();
    float        fps         = calcFPS(timeNowSec - lastTimeSec);
    string       prim        = _primitiveType == GL_TRIANGLES ? "GL_TRIANGLES" : "GL_TRIANGLE_STRIPS";
    snprintf(title,
             sizeof(title),
             "Sphere, %d x %d, fps: %4.0f, %s",
             _resolution,
             _resolution,
             fps,
             prim.c_str());
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
    _projectionMatrix.perspective(50.0f, (float)(w / h), 0.01f, 10.0f);

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
        _deltaY = (int)(_startX - x);
        _deltaX = (int)(_startY - y);
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
/*! Inits OpenGL and the GLFW window library
 */
void initGLFW(int wndWidth, int wndHeight, const char* wndTitle)
{
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

    // Create the GLFW window
    window = glfwCreateWindow(wndWidth,
                              wndHeight,
                              wndTitle,
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

    // Set GLFW callback functions
    glfwSetKeyCallback(window, onKey);
    glfwSetFramebufferSizeCallback(window, onResize);
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetCursorPosCallback(window, onMouseMove);
    glfwSetScrollCallback(window, onMouseWheel);
    glfwSetWindowCloseCallback(window, onClose);

    // Set number of monitor refreshes between 2 buffer swaps
    glfwSwapInterval(1);
}
//-----------------------------------------------------------------------------
/*!
The C main procedure running the GLFW GUI application.
*/
int main(int argc, char* argv[])
{
    _projectRoot = SLstring(SL_PROJECT_ROOT);

    _scrWidth  = 640;
    _scrHeight = 480;

    // Init OpenGL and the window library GLFW
    initGLFW(_scrWidth, _scrHeight, "ColorSphere");

    // Check errors before we start
    GETGLERROR;

    // Print OpenGL info on console
    glUtils::printGLInfo();

    // Prepare all our OpenGL stuff
    onInit();

    // Call once resize to define the projection
    onResize(window, _scrWidth, _scrHeight);

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
