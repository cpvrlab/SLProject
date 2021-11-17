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
    float life; //

    Particle()
      : p(0.0f), v(0.0f), c(1.0f),s(1.0f), life(0.0f) {}
};

struct ParticleNew
{
    SLVec3f p;    // particle position [x,y,z]
    SLVec3f v;    // particle velocity [x,y,z]
    float   st;  // particle start time
    SLVec3f initV;    // particle velocity [x,y,z]

    ParticleNew()
      : p(0.0f), v(0.0f), st(0.0f), initV(0.0f) {}
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

static GLuint _vao[2]; //!< ID of the vertex array object
static GLuint _tfo[2];    // Transform feedback objects
static GLuint _vbo[2];    // ID of the vertex buffer object

static GLuint _numV = 0; //!< NO. of vertices
static GLuint _numI = 0; //!< NO. of vertex indexes for triangles

vector<Particle> particles;  //!< List of particles
const int AMOUNT = 500; //!< Amount of particles
static int     _drawBuf = 0; // Boolean to switch buffer
static float     _ttl = 5.0f; // Boolean to switch buffer
static float     _currentTime = 0.0f; // Boolean to switch buffer
static float     _lastTime = 0.0f; // Boolean to switch buffer
static bool      _firstTimeRendering = true;

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

static GLuint _tDshaderVertID = 0; //! transform feedback vertex shader id
static GLuint _tDshaderFragID   = 0; //! transform feedback fragment shader id
static GLuint _tDshaderProgID = 0; //! transform feedback shader program id

// Attribute & uniform variable location indexes
static GLint _pLoc;   //!< attribute location for vertex position
static GLint _cLoc;   //!< attribute location for vertex color
static GLint _stLoc; //!< attribute location for vertex start time
static GLint _sLoc;   //!< uniform location for vertex scale
static GLint _tTLLoc; //!< uniform location for particle life time
static GLint _timeLoc; //!< uniform location for time
static GLint _tLoc;   //!< attribute location for vertex texture coord
static GLint _gLoc;   //!< uniform location for gamma value
static GLint _mvLoc; //!< uniform location for modelview matrix
static GLint _pMatLoc; //!< uniform location for projection matrix

// Attribute & uniform variable location indexes
static GLint _pTdLoc;    //!< attribute location for vertex position
static GLint _vTdLoc;    //!< attribute location for vertex velocity
static GLint _stTdLoc;    //!< attribute location for vertex start time
static GLint _initVTdLoc;    //!< attribute location for vertex initial velocity
static GLint _tTLTdLoc;  //!< uniform location for particle life time
static GLint _timeTdLoc; //!< uniform location for time 
static GLint _dTimeLoc; //!< uniform location for delta time
static GLint _aLoc; //!< uniform location for acceleration
static GLint _oTdLoc;     //!< uniform location for vertex offset

static GLint _texture0Loc; //!< uniform location for texture 0


float randomFloat(float a, float b)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff   = b - a;
    float r      = random * diff;
    return a + r;
}

//-----------------------------------------------------------------------------
void initParticles(float numberPerFrame, float timeToLive, SLVec3f offset)
{
    _ttl        = timeToLive;
    float* data = new GLfloat[AMOUNT * 10];
    int _count           = 0;
    ParticleNew p    = ParticleNew();
    p.p              = offset;
    p.v              = SLVec3f(0, 0.2, 0);
    p.initV          = p.v;

    for (unsigned int i = 0; i < AMOUNT * 10; i += 10)
    {
        _count++;
        p.v.x         = randomFloat(0.2f, -0.2f);
        p.v.y         = randomFloat(0.4f, 0.6f);
        p.st          = i * (timeToLive / (AMOUNT * 10)); // When the first particle dies the last one begin to live

        data[i] = p.p.x;
        data[i+1] = p.p.y;
        data[i+2] = p.p.z;

        data[i+3] = p.v.x;
        data[i+4] = p.v.y;
        data[i+5] = p.v.z;

        data[i+6] = p.v.z;

        data[i+7] = p.initV.x;
        data[i+8] = p.initV.y;
        data[i+9] = p.initV.z;
    }
    float _showOf = sizeof(data);
    glGenTransformFeedbacks(2, _tfo);
    glGenVertexArrays(2, _vao);
    glGenBuffers(2, _vbo);

    for (unsigned int i = 0; i < 2; i++)
    {
        glBindVertexArray(_vao[0]);
        glBindBuffer(GL_ARRAY_BUFFER, _vbo[i]);
        glBufferData(GL_ARRAY_BUFFER, (AMOUNT * sizeof(ParticleNew)), data, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleNew), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleNew), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleNew), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleNew), (void*)(7 * sizeof(float)));
        glBindVertexArray(0);

        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, _tfo[i]);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, _vbo[i]);
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

    // Load, compile & link shaders for transform feedback
    _tDshaderVertID = glUtils::buildShader(_projectRoot + "/data/shaders/ParticleTD.vert", GL_VERTEX_SHADER);
    _tDshaderFragID = glUtils::buildShader(_projectRoot + "/data/shaders/ParticleTD.frag", GL_FRAGMENT_SHADER);
    _tDshaderProgID = glUtils::buildProgramTD(_tDshaderVertID, _tDshaderFragID);

    // Load, compile & link shaders
    _shaderVertID = glUtils::buildShader(_projectRoot + "/data/shaders/Particle.vert", GL_VERTEX_SHADER);
    _shaderFragID = glUtils::buildShader(_projectRoot + "/data/shaders/Particle.frag", GL_FRAGMENT_SHADER);
    _shaderGeomID = glUtils::buildShader(_projectRoot + "/data/shaders/Particle.geom", GL_GEOMETRY_SHADER);
    _shaderProgID = glUtils::buildProgram(_shaderVertID,_shaderGeomID, _shaderFragID);

    // Activate the shader program
    glUseProgram(_tDshaderProgID);

    // Get the variable locations (identifiers) within the program
    /* _pTdLoc   = glGetAttribLocation(_tDshaderProgID, "a_position");
    _vTdLoc     = glGetAttribLocation(_tDshaderProgID, "a_velocity");
    _stTdLoc     = glGetAttribLocation(_tDshaderProgID, "a_startTime");
    _initVTdLoc = glGetAttribLocation(_tDshaderProgID, "a_initialVelocity");*/
    _tTLTdLoc   = glGetUniformLocation(_tDshaderProgID, "u_tTL");
    _timeTdLoc  = glGetUniformLocation(_tDshaderProgID, "u_time");
    _dTimeLoc = glGetUniformLocation(_tDshaderProgID, "u_deltaTime");
    _aLoc     = glGetUniformLocation(_tDshaderProgID, "u_acceleration");
    _oTdLoc        = glGetUniformLocation(_shaderProgID, "u_offset");

    // Activate the shader program
    glUseProgram(_shaderProgID);

    // Get the variable locations (identifiers) within the program
    /* _pLoc   = glGetAttribLocation(_shaderProgID, "a_position");
    _stLoc   = glGetAttribLocation(_shaderProgID, "a_startTime");*/
    _gLoc   = glGetUniformLocation(_shaderProgID, "u_oneOverGamma");
    _mvLoc = glGetUniformLocation(_shaderProgID, "u_mvMatrix");
    _pMatLoc     = glGetUniformLocation(_shaderProgID, "u_pMatrix");
    _cLoc        = glGetUniformLocation(_shaderProgID, "u_color");
    _sLoc        = glGetUniformLocation(_shaderProgID, "u_scale");
    _tTLLoc      = glGetUniformLocation(_shaderProgID, "u_tTL");
    _timeLoc     = glGetUniformLocation(_shaderProgID, "u_time");
    _texture0Loc = glGetUniformLocation(_shaderProgID, "u_matTextureDiffuse0");

    //buildBox();
    //buildSquare();
    initParticles(3.0f, 5.0f, SLVec3f(0, -0.5, 0));

    glClearColor(0.0f, 0.0f, 0.0f, 1); // Set the background color
    glEnable(GL_DEPTH_TEST);           // Enables depth test
    glEnable(GL_CULL_FACE);            // Enables the culling of back faces
    GETGLERROR;
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
    glDeleteShader(_shaderGeomID);
    glDeleteShader(_shaderFragID);
    glDeleteProgram(_shaderProgID);

    glDeleteShader(_tDshaderVertID);
    glDeleteShader(_tDshaderFragID);
    glDeleteProgram(_tDshaderProgID);

    // Delete arrays & buffers on GPU
    glDeleteVertexArrays(2, _vao);
    glDeleteBuffers(2, _vbo);
    glDeleteTransformFeedbacks(2, _tfo);
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

    _currentTime = glfwGetTime();
    float delatTime = _currentTime - _lastTime;
    _lastTime       = _currentTime;

    /////////// Update part ////////////////
    // Activate the shader program
    glUseProgram(_tDshaderProgID);
    // Set the uniforms for transform feedback shader
    glUniform1f(_tTLTdLoc, _ttl);
    glUniform1f(_timeTdLoc, _currentTime);
    glUniform3f(_aLoc, 1.0f, 1.0f, 1.0f);
    glUniform3f(_oTdLoc, 0.0f, -0.5f, 0.0f);
    glUniform1f(_dTimeLoc, delatTime);

    // Disable rendering
    glEnable(GL_RASTERIZER_DISCARD);
    // Bind the feedback object for the buffers to be drawn next
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,_tfo[_drawBuf]);
    // Draw points from input buffer with transform feedback
    glBeginTransformFeedback(GL_POINTS);

    glBindVertexArray(_vao[1 - _drawBuf]);
    /*if (_firstTimeRendering)
    {
        glDrawArrays(GL_POINTS, 0, AMOUNT); // Must use glDrawArrays the first time to specify number of points
        _firstTimeRendering = false;
    }
    else
    {
        // This uses the size of the Transform Feedback data to specify the number of points .
        // Apart from that, the next line is equivalent to using glDrawArrays(GL_POINTS, 0, ).
        glDrawTransformFeedback(GL_POINTS, _tfo[1 - _drawBuf]);
    }*/
    glDrawArrays(GL_POINTS, 0, AMOUNT);

    glEndTransformFeedback();
    // Enable rendering
    glDisable(GL_RASTERIZER_DISCARD);
    //////////// Render part ///////////////
    // Activate the shader program
    glUseProgram(_shaderProgID);

    //glEnable(GL_DEPTH_TEST);           // Enables depth test
    //glEnable(GL_CULL_FACE);            // Enables the culling of back faces

    // Activate Texture
    glBindTexture(GL_TEXTURE_2D, _textureID);

    glClear(GL_COLOR_BUFFER_BIT);
    // Initialize uniforms for transformation matrices if needed
    glUniformMatrix4fv(_mvLoc, 1, 0, (float*)&mv);
    glUniformMatrix4fv(_pMatLoc, 1, 0, (float*)&_projectionMatrix);
    glUniform1f(_gLoc, 1.0f);
    glUniform1i(_texture0Loc, 0);
    glUniform1f(_tTLLoc, _ttl); 
    glUniform1f(_timeLoc, _currentTime);
    glUniform1f(_sLoc, 1.0f);
    glUniform4f(_cLoc, 1.0f,1.0f,1.0f,1.0f);

     // Un-bind the feedback object.
     glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    // Draw the sprites from the feedback buffer
    glBindVertexArray(_vao[_drawBuf]);

    glEnable(GL_BLEND); // activate transparency (blending)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // use additive blending to give it a 'glow' effect

    glDrawArrays(GL_POINTS, 0, AMOUNT);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // don't forget to reset to default blending mode
    glDisable(GL_BLEND);                               // disable transparency

    // Swap buffers
    _drawBuf = 1 - _drawBuf;

    //////////// End  ///////////////


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
    //glfwGetPrimaryMonitor()
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
        /* currentFrame = glfwGetTime();
        deltaTime    = currentFrame - lastFrame;
        lastFrame    = currentFrame;
        timingRespawn += deltaTime;
        if (timingRespawn > 0.05f) {
            spawnParticles(3, SLVec3f(0, -0.5, 0));
            timingRespawn = 0.0f;
        }
        updateParticles(deltaTime);*/

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
