//#############################################################################
//  File:      ColorCube.cpp
//  Purpose:   Core profile OpenGL application with a colored cube with
//             GLFW as the OS GUI interface (http://www.glfw.org/).
//  Author:    Marcus Hudritsch
//  Date:      April 2016 (FS16)
//  Copyright: Marcus Hudritsch, Switzerland
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <GL/glew.h>    // OpenGL headers
#include <GLFW/glfw3.h> // GLFW GUI library
#include <SL.h>         // Basic SL type definitions
#include <SLMat4.h>     // 4x4 matrix class
#include <SLVec3.h>     // 3D vector class
#include <glUtils.h>    // Basics for OpenGL shaders, buffers & textures

//-----------------------------------------------------------------------------
//! Struct definition for vertex attribute position and color
struct VertexPC
{
    SLVec3f p; // vertex position [x,y,z]
    SLCol3f c; // vertex color [r,g,b]

    // Setter method
    void set(float posX,
             float posY,
             float posZ,
             float colorR,
             float colorG,
             float colorB)
    {
        p.set(posX, posY, posZ);
        c.set(colorR, colorG, colorB);
    }
};
//-----------------------------------------------------------------------------
// GLobal application variables
static GLFWwindow* window;       //!< The global GLFW window handle
static SLstring    _projectRoot; //!< Directory of executable
static SLint       _scrWidth;    //!< Window width at start up
static SLint       _scrHeight;   //!< Window height at start up
static SLfloat     _scr2fbX;     //!< Factor from screen to framebuffer coords
static SLfloat     _scr2fbY;     //!< Factor from screen to framebuffer coords

static SLMat4f _viewMatrix;       //!< 4x4 view matrix
static SLMat4f _modelMatrix;      //!< 4x4 model matrix
static SLMat4f _projectionMatrix; //!< 4x4 projection matrix

static GLuint _vao  = 0; //!< ID of the vertex array object
static GLuint _vboV = 0; //!< ID of the VBO for vertex attributes
static GLuint _vboI = 0; //!< ID of the VBO for vertex index array

static GLuint _numV = 0; //!< NO. of vertices
static GLuint _numI = 0; //!< NO. of vertex indexes for triangles

static float  _camZ;                   //!< z-distance of camera
static float  _rotX, _rotY;            //!< rotation angles around x & y axis
static int    _deltaX, _deltaY;        //!< delta mouse motion
static int    _startX, _startY;        //!< x,y mouse start positions
static int    _mouseX, _mouseY;        //!< current mouse position
static bool   _mouseLeftDown;          //!< Flag if mouse is down
static GLuint _modifiers = 0;          //!< modifier bit flags
const GLuint  NONE       = 0;          //!< constant for no modifier
const GLuint  SHIFT      = 0x00200000; //!< constant for shift key modifier
const GLuint  CTRL       = 0x00400000; //!< constant for control key modifier
const GLuint  ALT        = 0x00800000; //!< constant for alt key modifier

static GLuint _shaderVertID = 0; //! vertex shader id
static GLuint _shaderFragID = 0; //! fragment shader id
static GLuint _shaderProgID = 0; //! shader program id

// Attribute & uniform variable location indexes
static GLint _pLoc;   //!< attribute location for vertex position
static GLint _cLoc;   //!< attribute location for vertex color
static GLint _gLoc;   //!< uniform location for gamma value
static GLint _mvpLoc; //!< uniform location for modelview-projection matrix

//-----------------------------------------------------------------------------
void buildBox()
{
    // create C arrays on heap
    // Define the vertex position and colors as an array of structure
    // We define the colors with the same components as the cubes corners.
    _numV              = 8;
    VertexPC* vertices = new VertexPC[_numV];
    vertices[0].set(1, 1, 1, 1, 1, 1); //LTN
    vertices[1].set(1, 0, 1, 1, 0, 1); //LBN
    vertices[2].set(1, 0, 0, 1, 0, 0); //LBF
    vertices[3].set(1, 1, 0, 1, 1, 0); //LTF
    vertices[4].set(0, 0, 0, 0, 0, 0); //RBF
    vertices[5].set(0, 0, 1, 0, 0, 1); //RBN
    vertices[6].set(0, 1, 1, 0, 1, 1); //RTN
    vertices[7].set(0, 1, 0, 0, 1, 0); //RTF

    // Define the triangle indexes of the cubes vertices
    _numI           = 36;
    GLuint* indices = new GLuint[_numI];
    int     n       = 0;
    indices[n++]    = 0;
    indices[n++]    = 1;
    indices[n++]    = 2;
    indices[n++]    = 0;
    indices[n++]    = 2;
    indices[n++]    = 3; // Right
    indices[n++]    = 4;
    indices[n++]    = 5;
    indices[n++]    = 6;
    indices[n++]    = 4;
    indices[n++]    = 6;
    indices[n++]    = 7; // Left
    indices[n++]    = 0;
    indices[n++]    = 3;
    indices[n++]    = 7;
    indices[n++]    = 0;
    indices[n++]    = 7;
    indices[n++]    = 6; // Top
    indices[n++]    = 1;
    indices[n++]    = 5;
    indices[n++]    = 2;
    indices[n++]    = 2;
    indices[n++]    = 5;
    indices[n++]    = 4; // Bottom
    indices[n++]    = 0;
    indices[n++]    = 5;
    indices[n++]    = 1;
    indices[n++]    = 0;
    indices[n++]    = 6;
    indices[n++]    = 5; // Near
    indices[n++]    = 4;
    indices[n++]    = 7;
    indices[n++]    = 3;
    indices[n++]    = 3;
    indices[n++]    = 2;
    indices[n++]    = 4; // Far

    // Generate the OpenGL vertex array object
    glUtils::buildVAO(_vao,
                      _vboV,
                      _vboI,
                      vertices,
                      (GLint)_numV,
                      sizeof(VertexPC),
                      indices,
                      (GLint)_numI,
                      sizeof(GL_UNSIGNED_INT),
                      (GLint)_shaderProgID,
                      _pLoc,
                      _cLoc);

    // delete data on heap. The VBOs are now on the GPU
    delete[] vertices;
    delete[] indices;
}
//-----------------------------------------------------------------------------
/*!
onInit initializes the global variables and builds the shader program. It
should be called after a window with a valid OpenGL context is present.
*/
void onInit()
{
    // backwards movement of the camera
    _camZ = -3;

    // Mouse rotation parameters
    _rotX = _rotY = 0;
    _deltaX = _deltaY = 0;
    _mouseLeftDown    = false;

    // Load, compile & link shaders
    _shaderVertID = glUtils::buildShader(_projectRoot + "/data/shaders/ColorAttribute.vert", GL_VERTEX_SHADER);
    _shaderFragID = glUtils::buildShader(_projectRoot + "/data/shaders/Color.frag", GL_FRAGMENT_SHADER);
    _shaderProgID = glUtils::buildProgram(_shaderVertID, _shaderFragID);

    // Activate the shader program
    glUseProgram(_shaderProgID);

    // Get the variable locations (identifiers) within the program
    _pLoc   = glGetAttribLocation(_shaderProgID, "a_position");
    _cLoc   = glGetAttribLocation(_shaderProgID, "a_color");
    _gLoc   = glGetUniformLocation(_shaderProgID, "u_oneOverGamma");
    _mvpLoc = glGetUniformLocation(_shaderProgID, "u_mvpMatrix");

    buildBox();

    glClearColor(0.5f, 0.5f, 0.5f, 1); // Set the background color
    glEnable(GL_DEPTH_TEST);           // Enables depth test
    glEnable(GL_CULL_FACE);            // Enables the culling of back faces
    GETGLERROR;
}
//-----------------------------------------------------------------------------
/*!
onClose is called when the user closes the window and can be used for proper
deallocation of resources.
*/
void onClose(GLFWwindow* window)
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
    //1) Clear the color & depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //2a) View transform: move the coordinate system away from the camera
    _viewMatrix.identity();
    _viewMatrix.translate(0, 0, _camZ);

    //2b) Model transform: rotate the coordinate system increasingly
    _viewMatrix.rotate(_rotX + _deltaX, 1, 0, 0);
    _viewMatrix.rotate(_rotY + _deltaY, 0, 1, 0);

    //3) Model transform: move the cube so that it rotates around its center
    _modelMatrix.identity();
    _modelMatrix.translate(-0.5f, -0.5f, -0.5f);

    //4) Build the combined modelview-projection matrix
    SLMat4f mvp(_projectionMatrix);
    SLMat4f mv(_viewMatrix);
    mv.multiply(_modelMatrix);
    mvp.multiply(mv);

    //6) Activate the shader program and pass the uniform variables to the shader
    glUseProgram(_shaderProgID);
    glUniformMatrix4fv(_mvpLoc, 1, 0, (float*)&mvp);
    glUniform1f(_gLoc, 1.0f);

    //7a) Activate the vertex array
    glBindVertexArray(_vao);

    //7b) Activate the index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _vboI);

    //7c) Draw cube with triangles by indexes
    glDrawElements(GL_TRIANGLES, (GLint)_numI, GL_UNSIGNED_INT, nullptr);

    //8) Fast copy the back buffer to the front buffer. This is OS dependent.
    glfwSwapBuffers(window);
    GETGLERROR;

    // Return true to get an immediate refresh
    return true;
}
//-----------------------------------------------------------------------------
/*!
onResize: Event handler called on the resize event of the window. This event
should called once before the onPaint event. Do everything that is dependent on
the size and ratio of the window.
*/
void onResize(GLFWwindow* window, int width, int height)
{
    float w = (float)width;
    float h = (float)height;

    // define the projection matrix
    _projectionMatrix.perspective(50, w / h, 0.01f, 10.0f);

    // define the viewport
    glViewport(0, 0, width, height);

    GETGLERROR;

    onPaint();
}
//-----------------------------------------------------------------------------
/*!
Mouse button down & release eventhandler starts and end mouse rotation
*/
void onMouseButton(GLFWwindow* window, int button, int action, int mods)
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
void onMouseMove(GLFWwindow* window, double x, double y)
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
Mouse wheel eventhandler that moves the camera forward or backwards
*/
void onMouseWheel(GLFWwindow* window, double xscroll, double yscroll)
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
void onKey(GLFWwindow* window, int GLFWKey, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (GLFWKey)
        {
            case GLFW_KEY_ESCAPE:
                onClose(window);
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_LEFT_SHIFT: _modifiers = _modifiers | SHIFT; break;
            case GLFW_KEY_RIGHT_SHIFT: _modifiers = _modifiers | SHIFT; break;
            case GLFW_KEY_LEFT_CONTROL: _modifiers = _modifiers | CTRL; break;
            case GLFW_KEY_RIGHT_CONTROL: _modifiers = _modifiers | CTRL; break;
            case GLFW_KEY_LEFT_ALT: _modifiers = _modifiers | ALT; break;
            case GLFW_KEY_RIGHT_ALT: _modifiers = _modifiers | ALT; break;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        switch (GLFWKey)
        {
            case GLFW_KEY_LEFT_SHIFT: _modifiers = _modifiers ^ SHIFT; break;
            case GLFW_KEY_RIGHT_SHIFT: _modifiers = _modifiers ^ SHIFT; break;
            case GLFW_KEY_LEFT_CONTROL: _modifiers = _modifiers ^ CTRL; break;
            case GLFW_KEY_RIGHT_CONTROL: _modifiers = _modifiers ^ CTRL; break;
            case GLFW_KEY_LEFT_ALT: _modifiers = _modifiers ^ ALT; break;
            case GLFW_KEY_RIGHT_ALT: _modifiers = _modifiers ^ ALT; break;
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

//You can enable or restrict newer OpenGL context here (read the GLFW documentation)
#ifdef SL_OS_MACOS
//glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#else
//glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
//glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    _scrWidth  = 640;
    _scrHeight = 480;

    // Create the GLFW window
    window = glfwCreateWindow(_scrWidth,
                              _scrHeight,
                              "Color Cube",
                              nullptr,
                              nullptr);

    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Get the current GL context. After this you can call GL
    glfwMakeContextCurrent(window);

    // On some systems screen & framebuffer size are different (e.g. Mac Retina Displays)
    // All commands in GLFW are in screen coords but rendering in GL is
    // in framebuffer coords
    SLint fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    _scr2fbX = (float)fbWidth / (float)_scrWidth;
    _scr2fbY = (float)fbHeight / (float)_scrHeight;

    // Include OpenGL via GLEW (init must be after window creation)
    // The goal of the OpenGL Extension Wrangler Library (GLEW) is to assist C/C++
    // OpenGL developers with two tedious tasks: initializing and using extensions
    // and writing portable applications. GLEW provides an efficient run-time
    // mechanism to determine whether a certain extension is supported by the
    // driver or not. OpenGL core and extension functionality is exposed via a
    // single header file. Download GLEW at: http://glew.sourceforge.net/
    glewExperimental = GL_TRUE; // avoids a crash
    GLenum err       = glewInit();
    if (GLEW_OK != err)
    {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Check errors before we start
    GETGLERROR;

    glUtils::printGLInfo();

    // Set number of monitor refreshes between 2 buffer swaps
    glfwSwapInterval(1);

    // Prepare all OpenGL stuff
    onInit();

    // Call resize once for correct projection
    onResize(window, (SLint)(_scrWidth * _scr2fbX), (SLint)(_scrHeight * _scr2fbY));

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
