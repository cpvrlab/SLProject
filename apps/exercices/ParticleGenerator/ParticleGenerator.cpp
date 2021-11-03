//#############################################################################
//  File:      ParticleGenerator.cpp
//  Purpose:   Core profile OpenGL application with a colored cube with
//             GLFW as the OS GUI interface (http://www.glfw.org/).
//  Date:      October 2021
//  Authors:   Affolter Marc
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

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
//! Struct definition for particle attribute position, velocity, color and life
struct Particle
{
    SLVec3f p; // particle position [x,y,z]
    SLVec3f v; // particle velocity [x,y,z]
    SLCol4f c; // particle color [r,g,b,a]
    float s; // particle scale 
    float life;

    Particle()
      : p(0.0f), v(0.0f), c(1.0f),s(1.0f), life(0.0f) {}
};
//-----------------------------------------------------------------------------
// GLobal application variables
static GLFWwindow* window;       //!< The global GLFW window handle
static SLstring    _projectRoot; //!< Directory of executable
static SLint       _scrWidth;    //!< Window width at start up
static SLint       _scrHeight;   //!< Window height at start up

static SLMat4f _viewMatrix;       //!< 4x4 view matrix
static SLMat4f _modelMatrix;      //!< 4x4 model matrix
static SLMat4f _projectionMatrix; //!< 4x4 projection matrix

static GLuint _vao  = 0; //!< ID of the vertex array object
static GLuint _vboV = 0; //!< ID of the VBO for vertex attributes

static GLuint _numV = 0; //!< NO. of vertices
static GLuint _numI = 0; //!< NO. of vertex indexes for triangles

vector<Particle> particles;  //!< List of particles
const int AMOUNT = 500; //!< Amount of particles

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
static GLuint _shaderGeomID = 0; //! geometry shader id
static GLuint _shaderProgID = 0; //! shader program id
static GLuint _textureID    = 0; //!< texture id

// Attribute & uniform variable location indexes
static GLint _pLoc;   //!< attribute location for vertex position
static GLint _cLoc;   //!< attribute location for vertex color
static GLint _oLoc;   //!< attribute location for vertex offset
static GLint _crLoc;   //!< uniform location for camera right
static GLint _cuLoc;   //!< uniform location for camera up
static GLint _sLoc;   //!< uniform location for vertex scale
static GLint _tLoc;   //!< attribute location for vertex texture coord
static GLint _gLoc;   //!< uniform location for gamma value
static GLint _mvLoc; //!< uniform location for modelview matrix
static GLint _pMatLoc; //!< uniform location for projection matrix

static GLint _texture0Loc; //!< uniform location for texture 0

//-----------------------------------------------------------------------------
void initParticles()
{
    float points[] = {
      0.0f, 0.0f //
    };
    glGenBuffers(1, &_vboV);
    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vboV);
    glBufferData(GL_ARRAY_BUFFER, sizeof(points), &points, GL_STATIC_DRAW);
    glEnableVertexAttribArray(_pLoc);
    glVertexAttribPointer(_pLoc, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glBindVertexArray(0);

    // create this->amount default particle instances
    for(unsigned int i = 0; i < AMOUNT; ++i)
    {
        particles.push_back(Particle());
    }
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

    // Load textures
    _textureID = glUtils::buildTexture(_projectRoot + "/data/images/textures/circle_01.png");


    // Load, compile & link shaders
    _shaderVertID = glUtils::buildShader(_projectRoot + "/data/shaders/Particle.vert", GL_VERTEX_SHADER);
    _shaderFragID = glUtils::buildShader(_projectRoot + "/data/shaders/Particle.frag", GL_FRAGMENT_SHADER);
    _shaderGeomID = glUtils::buildShader(_projectRoot + "/data/shaders/Particle.geom", GL_GEOMETRY_SHADER);
    _shaderProgID = glUtils::buildProgram(_shaderVertID,_shaderGeomID, _shaderFragID);

    // Activate the shader program
    glUseProgram(_shaderProgID);

    // Get the variable locations (identifiers) within the program
    _pLoc   = glGetAttribLocation(_shaderProgID, "a_position");
    _gLoc   = glGetUniformLocation(_shaderProgID, "u_oneOverGamma");
    _mvLoc = glGetUniformLocation(_shaderProgID, "u_mvMatrix");
    _pMatLoc     = glGetUniformLocation(_shaderProgID, "u_pMatrix");
    _cLoc        = glGetUniformLocation(_shaderProgID, "u_color");
    _oLoc        = glGetUniformLocation(_shaderProgID, "u_offset");
    _sLoc        = glGetUniformLocation(_shaderProgID, "u_scale");
    _texture0Loc = glGetUniformLocation(_shaderProgID, "u_matTextureDiffuse0");

    //buildBox();
    //buildSquare();
    initParticles();

    glClearColor(0.0f, 0.0f, 0.0f, 1); // Set the background color
    glEnable(GL_DEPTH_TEST);           // Enables depth test
    glEnable(GL_CULL_FACE);            // Enables the culling of back faces
    GETGLERROR;
}

// stores the index of the last particle used (for quick access to next dead particle)
unsigned int lastUsedParticle = 0;
unsigned int firstUnusedParticle()
{
    // first search from last used particle, this will usually return almost instantly
    for (unsigned int i = lastUsedParticle; i < AMOUNT; ++i)
    {
        if (particles[i].life <= 0.0f)
        {
            lastUsedParticle = i;
            return i;
        }
    }
    // otherwise, do a linear search
    for (unsigned int i = 0; i < lastUsedParticle; ++i)
    {
        if (particles[i].life <= 0.0f)
        {
            lastUsedParticle = i;
            return i;
        }
    }
    // all particles are taken, override the first one (note that if it repeatedly hits this case, more particles should be reserved)
    lastUsedParticle = 0;
    return 0;
}

float randomFloat(float a, float b)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff   = b - a;
    float r      = random * diff;
    return a + r;
}

void respawnParticle(Particle& particle, SLVec3f offset)
{
    float random      = (rand() % 100) / 100.0f;
    float rColor      = 0.5f + ((rand() % 100) / 100.0f);
    particle.p =  offset;
    //particle.p.x += random;
    particle.c    = SLVec4f(rColor, rColor, rColor, 1.0f);
    particle.life     = 5.0f;
    particle.s        = randomFloat(0.5f, 1.0f);
    //particle.s        = 0.5f;
    particle.v        = SLVec3f(0, 0.2, 0);
    particle.v.x      = randomFloat(0.2f, -0.2f);
    particle.v.y      = randomFloat(0.4f, 0.6f);
    
}
void spawnParticles(unsigned int newParticles, SLVec3f offset)
{
    // add new particles
    for (unsigned int i = 0; i < newParticles; ++i)
    {
        int unusedParticle = firstUnusedParticle();
        respawnParticle(particles[unusedParticle], offset);
    }
}

void updateParticles(float dt)
{
    // update all particles
    for (unsigned int i = 0; i < AMOUNT; ++i)
    {
        Particle& p = particles[i];
        p.life-= dt; // reduce life
        if (p.life > 0.0f)
        { // particle is alive, thus update
            p.p += p.v * dt;
            p.c.a -= dt * 0.25f;
        }
    }
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
    //_modelMatrix.translate(-0.5f, -0.5f, -0.5f);

    //4) Build the combined modelview-projection matrix
    SLMat4f mv(_viewMatrix);
    mv.multiply(_modelMatrix);

    //6) Activate the shader program and pass the uniform variables to the shader
    glUseProgram(_shaderProgID);
    glUniformMatrix4fv(_mvLoc, 1, 0, (float*)&mv);
    glUniformMatrix4fv(_pMatLoc, 1, 0, (float*)&_projectionMatrix);
    glUniform1f(_gLoc, 1.0f);
    glUniform1i(_texture0Loc, 0);

    //7a) Activate the vertex array
    glBindVertexArray(_vao);

    // Activate Texture
    glBindTexture(GL_TEXTURE_2D, _textureID);
    
    glEnable(GL_BLEND); // activate transparency (blending)
    // use additive blending to give it a 'glow' effect
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    int count = 0;
    for (Particle particle : particles)
    {
        count++;
        if (particle.life > 0.0f)
        {
            glUniform1f(_sLoc, particle.s);
            glUniform4f(_cLoc, particle.c.r, particle.c.g, particle.c.b, particle.c.a);
            glUniform3f(_oLoc, particle.p.x, particle.p.y, particle.p.z);
            //7c) Draw cube with triangles by indexes
            glDrawArrays(GL_POINTS, 0, 1);
            
        }
    }
    // don't forget to reset to default blending mode
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND); // disable transparency

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
#ifdef __APPLE__
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GL_FALSE);
#else
    //glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    //glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    //glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    _scrWidth  = 2560;
    _scrHeight = 1440;

    // Create the GLFW window
    window = glfwCreateWindow(_scrWidth,
                              _scrHeight,
                              "Particle Generator",
                              glfwGetPrimaryMonitor(),
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
    float lastFrame = glfwGetTime();
    float currentFrame = 0.0f;
    float deltaTime = 0.0f;
    float timingRespawn = 0.0f;
    while (!glfwWindowShouldClose(window))
    {
        currentFrame = glfwGetTime();
        deltaTime    = currentFrame - lastFrame;
        lastFrame    = currentFrame;
        timingRespawn += deltaTime;
        if (timingRespawn > 0.05f) {
            spawnParticles(3, SLVec3f(0, -0.5, 0));
            timingRespawn = 0.0f;
        }
        updateParticles(deltaTime);

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
