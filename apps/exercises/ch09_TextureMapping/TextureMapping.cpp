//#############################################################################
//  File:      TextureMapping.cpp
//  Purpose:   Minimal core profile OpenGL application for ambient-diffuse-
//             specular lighting shaders with Textures.
//  Date:      February 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <Utils.h>
#include <GL/gl3w.h>    // OpenGL headers
#include <GLFW/glfw3.h> // GLFW GUI library
#include <SLMat4.h>     // 4x4 matrix class
#include <SLVec3.h>     // 3D vector class
#include <glUtils.h>    // Basics for OpenGL shaders, buffers & textures

//-----------------------------------------------------------------------------
//! Struct definition for vertex attributes
struct VertexPNT
{
    SLVec3f p; // vertex position [x,y,z]
    SLVec3f n; // vertex normal [x,y,z]
    SLVec2f t; // vertex texture coord. [s,t]
};
//-----------------------------------------------------------------------------
static GLFWwindow* window;       //!< The global glfw window handle
static SLstring    _projectRoot; //!< Directory of executable
static SLint       _scrWidth;    //!< Window width at start up
static SLint       _scrHeight;   //!< Window height at start up

// Global application variables
static SLMat4f _cameraMatrix;                //!< 4x4 matrix for camera to world transform
static SLMat4f _viewMatrix;                  //!< 4x4 matrix for world to camera transform
static SLMat4f _modelMatrix;                 //!< 4x4 matrix for model to world transform
static SLMat4f _lightMatrix;                 //!< 4x4 matrix for light to world transform
static SLMat4f _projectionMatrix;            //!< Projection from view space to normalized device coordinates

static GLuint _vao  = 0;                     //!< ID of the vertex array object
static GLuint _vboV = 0;                     //!< ID of the VBO for vertex attributes
static GLuint _vboI = 0;                     //!< ID of the VBO for vertex index array
static GLuint _numV = 0;                     //!< NO. of vertices
static GLuint _numI = 0;                     //!< NO. of vertex indexes for triangles

static GLint _resolution;                    //!< resolution of sphere stack & slices

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

static SLVec4f _globalAmbi;                  //!< global ambient intensity
static SLVec4f _lightAmbient;                //!< Light ambient intensity
static SLVec4f _lightDiffuse;                //!< Light diffuse intensity
static SLVec4f _lightSpecular;               //!< Light specular intensity
static float   _lightSpotDeg;                //!< Light spot cutoff angle in degrees
static float   _lightSpotExp;                //!< Light spot exponent
static SLVec3f _lightAtt;                    //!< Light attenuation factors
static SLVec4f _matAmbient;                  //!< Material ambient reflection coeff.
static SLVec4f _matDiffuse;                  //!< Material diffuse reflection coeff.
static SLVec4f _matSpecular;                 //!< Material specular reflection coeff.
static SLVec4f _matEmissive;                 //!< Material emissive coeff.
static float   _matShininess;                //!< Material shininess exponent

static GLuint _shaderVertID = 0;             //!< vertex shader id
static GLuint _shaderFragID = 0;             //!< fragment shader id
static GLuint _shaderProgID = 0;             //!< shader program id
static GLuint _textureID    = 0;             //!< texture id

static GLint _pLoc;                          //!< attribute location for vertex position
static GLint _nLoc;                          //!< attribute location for vertex normal
static GLint _uvLoc;                         //!< attribute location for vertex texcoords
static GLint _pmLoc;                         //!< uniform location for projection matrix
static GLint _vmLoc;                         //!< uniform location for view matrix
static GLint _mmLoc;                         //!< uniform location for model matrix
static GLint _globalAmbiLoc;                 //!< uniform location for global ambient intensity
static GLint _lightPosVSLoc;                 //!< uniform location for light position in VS
static GLint _lightSpotDirVSLoc;             //!< uniform location for light direction in VS
static GLint _lightAmbientLoc;               //!< uniform location for ambient light intensity
static GLint _lightDiffuseLoc;               //!< uniform location for diffuse light intensity
static GLint _lightSpecularLoc;              //!< uniform location for specular light intensity
static GLint _lightAttLoc;                   //!< uniform location fpr light attenuation factors
static GLint _matAmbientLoc;                 //!< uniform location for ambient light reflection
static GLint _matDiffuseLoc;                 //!< uniform location for diffuse light reflection
static GLint _matSpecularLoc;                //!< uniform location for specular light reflection
static GLint _matEmissiveLoc;                //!< uniform location for light emission
static GLint _matShininessLoc;               //!< uniform location for shininess
static GLint _matTexDiffLoc;                 //!< uniform location for texture 0

//-----------------------------------------------------------------------------
/*!
buildSphere creates the vertex attributes for a sphere and creates the VBO
at the end. The sphere is built in stacks & slices. The slices are around the
z-axis.
*/
void buildSphere(float radius, GLuint stacks, GLuint slices)
{
    assert(stacks > 3 && slices > 3);

    // create vertex array
    GLuint     numV = (stacks + 1) * (slices + 1);
    VertexPNT* v    = new VertexPNT[numV];

    float  theta, dtheta; // angles around x-axis
    float  phi, dphi;     // angles around z-axis
    float  s, t, ds, dt;  // texture coords
    int    i, j;          // loop counters
    GLuint iv = 0;

    // init start values
    theta  = 0.0f;
    dtheta = Utils::PI / stacks;
    dphi   = 2.0f * Utils::PI / slices;
    ds     = 0.0f; // ???
    dt     = 0.0f; // ???
    t      = 0.0f; // ???

    // Define vertex position & normals by looping through all stacks
    for (i = 0; i <= (int)stacks; ++i)
    {
        float sin_theta = sin(theta);
        float cos_theta = cos(theta);
        phi = s = 0.0f;

        // Loop through all slices
        for (j = 0; j <= (int)slices; ++j)
        {
            if (j == (int)slices) phi = 0.0f;

            // define first the normal with length 1
            v[iv].n.x = sin_theta * cos(phi);
            v[iv].n.y = sin_theta * sin(phi);
            v[iv].n.z = cos_theta;

            // set the vertex position w. the scaled normal
            v[iv].p.x = radius * v[iv].n.x;
            v[iv].p.y = radius * v[iv].n.y;
            v[iv].p.z = radius * v[iv].n.z;

            // set the texture coords.
            v[iv].t.x = s;
            v[iv].t.y = t;

            phi += dphi;
            //s = ???
            iv++;
        }
        theta += dtheta;
        //t = ???
    }

    // create index array for triangles
    _numI           = (GLuint)(slices * stacks * 2 * 3);
    GLuint* indices = new GLuint[_numI];
    GLuint  ii      = 0, iV1, iV2;

    for (i = 0; i < (int)stacks; ++i)
    {
        // index of 1st & 2nd vertex of stack
        iV1 = i * (slices + 1);
        iV2 = iV1 + slices + 1;

        for (j = 0; j < (int)slices; ++j)
        {
            // 1st triangle ccw
            indices[ii++] = iV1 + j;
            indices[ii++] = iV2 + j;
            indices[ii++] = iV2 + j + 1;
            // 2nd triangle ccw
            indices[ii++] = iV1 + j;
            indices[ii++] = iV2 + j + 1;
            indices[ii++] = iV1 + j + 1;
        }
    }

    // Generate the OpenGL vertex array object
    glUtils::buildVAO(_vao,
                      _vboV,
                      _vboI,
                      v,
                      (GLint)numV,
                      sizeof(VertexPNT),
                      indices,
                      (GLint)_numI,
                      sizeof(GL_UNSIGNED_INT),
                      (GLint)_shaderProgID,
                      _pLoc,
                      -1,
                      _nLoc,
                      _uvLoc);

    // Delete arrays on heap
    delete[] v;
    delete[] indices;
}
//-----------------------------------------------------------------------------
/*!
buildSquare creates the vertex attributes for a textured square and VBO.
The square lies in the x-z-plane and is facing towards -y (downwards).
*/
void buildSquare()
{
    // create vertex array for interleaved position, normal and texCoord
    //                  Position,  Normal, texCrd,
    _numV = 4;

    // clang-format off
    float vertices[] = {-1, 0,-1, 0,-1, 0, 0, 0, // Vertex 0
                         1, 0,-1, 0,-1, 0, 1, 0, // Vertex 1
                         1, 0, 1, 0,-1, 0, 1, 1, // Vertex 2
                        -1, 0, 1, 0,-1, 0, 0, 1}; // Vertex 3
    // clang-format on
    // create index array for GL_TRIANGLES
    _numI            = 6;
    GLuint indices[] = {0, 1, 2, 0, 2, 3};

    // Generate the OpenGL vertex array object
    glUtils::buildVAO(_vao,
                      _vboV,
                      _vboI,
                      vertices,
                      (GLint)_numV,
                      sizeof(VertexPNT),
                      indices,
                      (GLint)_numI,
                      sizeof(GL_UNSIGNED_INT),
                      (GLint)_shaderProgID,
                      _pLoc,
                      _nLoc,
                      _uvLoc);

    // The vertices and indices are on the stack memory and get deleted at the
    // end of the block.
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

    for (SLuint i = 0; i < FILTERSIZE; ++i)
        sumTime += frameTimes[i];

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
    // Set light parameters
    _globalAmbi.set(0.0f, 0.0f, 0.0f);
    _lightAmbient.set(0.1f, 0.1f, 0.1f);
    _lightDiffuse.set(1.0f, 1.0f, 1.0f);
    _lightSpecular.set(1.0f, 1.0f, 1.0f);
    _lightMatrix.translate(0, 0, 3);
    _lightSpotDeg = 180.0f;           // point light
    _lightSpotExp = 1.0f;
    _lightAtt     = SLVec3f(1, 0, 0); // constant light attenuation = no attenuation
    _matAmbient.set(1.0f, 1.0f, 1.0f);
    _matDiffuse.set(1.0f, 1.0f, 1.0f);
    _matSpecular.set(1.0f, 1.0f, 1.0f);
    _matEmissive.set(0.0f, 0.0f, 0.0f);
    _matShininess = 500.0f;

    // position of the camera
    _camZ = 3.0f;

    // Mouse rotation parameters
    _rotX          = 0;
    _rotY          = 0;
    _deltaX        = 0;
    _deltaY        = 0;
    _mouseLeftDown = false;

    // Load textures
    _textureID = glUtils::buildTexture(_projectRoot + "/data/images/textures/earth2048_C.png");

    // Load, compile & link shaders
    _shaderVertID = glUtils::buildShader(_projectRoot + "/data/shaders/ch09_TextureMapping.vert", GL_VERTEX_SHADER);
    _shaderFragID = glUtils::buildShader(_projectRoot + "/data/shaders/ch09_TextureMapping.frag", GL_FRAGMENT_SHADER);
    _shaderProgID = glUtils::buildProgram(_shaderVertID, _shaderFragID);

    // Activate the shader program
    glUseProgram(_shaderProgID);

    // Get the variable locations (identifiers) within the vertex & pixel shader programs
    _pLoc              = glGetAttribLocation(_shaderProgID, "a_position");
    _nLoc              = glGetAttribLocation(_shaderProgID, "a_normal");
    _uvLoc             = glGetAttribLocation(_shaderProgID, "a_uv");
    _pmLoc             = glGetUniformLocation(_shaderProgID, "u_pMatrix");
    _vmLoc             = glGetUniformLocation(_shaderProgID, "u_vMatrix");
    _mmLoc             = glGetUniformLocation(_shaderProgID, "u_mMatrix");
    _globalAmbiLoc     = glGetUniformLocation(_shaderProgID, "u_globalAmbi");
    _lightPosVSLoc     = glGetUniformLocation(_shaderProgID, "u_lightPosVS");
    _lightSpotDirVSLoc = glGetUniformLocation(_shaderProgID, "u_lightSpotDir");
    _lightAmbientLoc   = glGetUniformLocation(_shaderProgID, "u_lightAmbi");
    _lightDiffuseLoc   = glGetUniformLocation(_shaderProgID, "u_lightDiff");
    _lightSpecularLoc  = glGetUniformLocation(_shaderProgID, "u_lightSpec");
    _lightAttLoc       = glGetUniformLocation(_shaderProgID, "u_lightAtt");
    _matAmbientLoc     = glGetUniformLocation(_shaderProgID, "u_matAmbi");
    _matDiffuseLoc     = glGetUniformLocation(_shaderProgID, "u_matDiff");
    _matSpecularLoc    = glGetUniformLocation(_shaderProgID, "u_matSpec");
    _matEmissiveLoc    = glGetUniformLocation(_shaderProgID, "u_matEmis");
    _matShininessLoc   = glGetUniformLocation(_shaderProgID, "u_matShin");
    _matTexDiffLoc     = glGetUniformLocation(_shaderProgID, "u_matTexDiff");

    // Build object
    // buildSphere(1.0f, 72, 72);
    buildSquare();

    // Set some OpenGL states
    glClearColor(0.0f, 0.0f, 0.0f, 1); // Set the background color
    glEnable(GL_DEPTH_TEST);           // Enables depth test
    glEnable(GL_CULL_FACE);            // Enables the culling of back faces
    GETGLERROR;                        // Check for OpenGL errors
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
    glDeleteBuffers(1, &_vboV);
    glDeleteBuffers(1, &_vboI);
}
//-----------------------------------------------------------------------------
/*!
onPaint does all the rendering for one frame from scratch with OpenGL (in core
profile).
*/
bool onPaint()
{
    // 1) Clear the color & depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* 2a) Model transform: rotate the coordinate system increasingly
     * first around the y- and then around the x-axis. This type of camera
     * transform is called turntable animation.*/
    _cameraMatrix.identity();
    _cameraMatrix.rotate(_rotY + _deltaY, 0, 1, 0);
    _cameraMatrix.rotate(_rotX + _deltaX, 1, 0, 0);

    // 2b) Move the camera to its position.
    _cameraMatrix.translate(0, 0, _camZ);

    // 2c) View transform is world to camera (= inverse of camera matrix)
    _viewMatrix = _cameraMatrix.inverted();

    // 3a) Rotate the model so that we see the square from the front side or the earth from the equator.
    _modelMatrix.identity();
    _modelMatrix.rotate(90, -1, 0, 0);

    // 4a) Transform light position into view space
    SLVec4f lightPosVS = _viewMatrix * SLVec4f(_lightMatrix.translation());

    // 4b) The spotlight direction is down the negative z-axis of the light transform
    SLVec3f lightSpotDirVS = _viewMatrix.mat3() * -_lightMatrix.axisZ();

    // 5) Activate the shader program and pass the uniform variables to the shader
    glUseProgram(_shaderProgID);
    glUniformMatrix4fv(_pmLoc, 1, 0, (float*)&_projectionMatrix);
    glUniformMatrix4fv(_vmLoc, 1, 0, (float*)&_viewMatrix);
    glUniformMatrix4fv(_mmLoc, 1, 0, (float*)&_modelMatrix);
    glUniform4fv(_globalAmbiLoc, 1, (float*)&_globalAmbi);
    glUniform3fv(_lightPosVSLoc, 1, (float*)&lightPosVS);
    glUniform3fv(_lightSpotDirVSLoc, 1, (float*)&lightSpotDirVS);
    glUniform4fv(_lightAmbientLoc, 1, (float*)&_lightAmbient);
    glUniform4fv(_lightDiffuseLoc, 1, (float*)&_lightDiffuse);
    glUniform4fv(_lightSpecularLoc, 1, (float*)&_lightSpecular);
    glUniform3fv(_lightAttLoc, 1, (float*)&_lightAtt);
    glUniform4fv(_matAmbientLoc, 1, (float*)&_matAmbient);
    glUniform4fv(_matDiffuseLoc, 1, (float*)&_matDiffuse);
    glUniform4fv(_matSpecularLoc, 1, (float*)&_matSpecular);
    glUniform4fv(_matEmissiveLoc, 1, (float*)&_matEmissive);
    glUniform1f(_matShininessLoc, _matShininess);
    glUniform1i(_matTexDiffLoc, 0);

    // 6) Activate the vertex array
    glBindVertexArray(_vao);

    // 7) Draw model triangles by indexes
    glDrawElements(GL_TRIANGLES, (GLsizei)_numI, GL_UNSIGNED_INT, nullptr);

    // 8) Fast copy the back buffer to the front buffer. This is OS dependent.
    glfwSwapBuffers(window);

    // 9) Check for OpenGL errors (only in debug done)
    GETGLERROR;

    // Calculate frames per second
    char         title[255];
    static float lastTimeSec = 0;
    float        timeNowSec  = (float)glfwGetTime();
    float        fps         = calcFPS(timeNowSec - lastTimeSec);
    snprintf(title, sizeof(title), "Texture Mapping %3.1f", fps);
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
    float w = (float)width;
    float h = (float)height;

    // define the projection matrix
    _projectionMatrix.perspective(45,
                                  w / h,
                                  0.01f,
                                  10.0f);

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
            case GLFW_KEY_UP:
                _resolution = _resolution << 1;
                buildSphere(1.0f, (GLuint)_resolution, (GLuint)_resolution);
                break;
            case GLFW_KEY_DOWN:
                if (_resolution > 4) _resolution = _resolution >> 1;
                buildSphere(1.0f, (GLuint)_resolution, (GLuint)_resolution);
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
    initGLFW(_scrWidth, _scrHeight, "TextureMapping");

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
        // if no updated occured wait for the next event (power saving)
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
