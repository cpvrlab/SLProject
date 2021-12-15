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
//! Struct definition for particle attribute position, velocity, start time and initial velocity
struct Particle
{
    SLVec3f p;    // particle position [x,y,z]
    SLVec3f v;    // particle velocity [x,y,z]
    float   st;  // particle start time
    SLVec3f initV;    // particle initial velocity [x,y,z]

    Particle()
      : p(0.0f), v(0.0f), st(0.0f), initV(0.0f) {}
};
//! Struct definition for vertex attribute position and color for Cube
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

static SLMat4f _viewMatrix;       //!< 4x4 view matrix
static SLMat4f _modelMatrix;      //!< 4x4 model matrix
static SLMat4f _projectionMatrix; //!< 4x4 projection matrix

static GLuint _vao[2]; //!< ID of the vertex array object
static GLuint _tfo[2];    // Transform feedback objects
static GLuint _vbo[2];    // ID of the vertex buffer object

//For cube rendering
static GLuint _vaoC  = 0; //!< ID of the vertex array object for the Cube
static GLuint _vboVC = 0; //!< ID of the VBO for vertex attributes for the Cube
static GLuint _vboIC = 0; //!< ID of the VBO for vertex index array for the Cube

static GLuint _numV = 0; //!< NO. of vertices
static GLuint _numI = 0; //!< NO. of vertex indexes for triangles

const int AMOUNT = 500; //!< Amount of particles
static int     _drawBuf = 0; // Boolean to switch buffer
static float     _ttl = 5.0f; // Boolean to switch buffer
static float     _currentTime = 0.0f; // Boolean to switch buffer
static float     _lastTime = 0.0f; // Boolean to switch buffer

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

static GLuint _cShaderVertID = 0; //! vertex cube shader id
static GLuint _cShaderFragID = 0; //! fragment cube shader id
static GLuint _cShaderProgID = 0; //! shader cube program id

static GLuint _tFShaderVertID = 0; //! transform feedback vertex shader id
static GLuint _tFShaderFragID   = 0; //! transform feedback fragment shader id
static GLuint _tFShaderProgID = 0; //! transform feedback shader program id

// Attribute & uniform variable location indexes
static GLint _pLoc;   //!< attribute location for vertex position
static GLint _cLoc;   //!< attribute location for vertex color
static GLint _stLoc; //!< attribute location for vertex start time
static GLint _sLoc;   //!< uniform location for vertex scale
static GLint _radiusLoc;   //!< uniform location for particle radius
static GLint _tTLLoc; //!< uniform location for particle life time
static GLint _timeLoc; //!< uniform location for time
static GLint _tLoc;   //!< attribute location for vertex texture coord
static GLint _gLoc;   //!< uniform location for gamma value
static GLint _mvLoc; //!< uniform location for modelview matrix
static GLint _pMatLoc; //!< uniform location for projection matrix

// Attribute & uniform variable location indexes
static GLint _pTFLoc;    //!< attribute location for vertex position
static GLint _vTFLoc;    //!< attribute location for vertex velocity
static GLint _stTFLoc;    //!< attribute location for vertex start time
static GLint _initVTFLoc;    //!< attribute location for vertex initial velocity
static GLint _tTLTFLoc;  //!< uniform location for particle life time
static GLint _timeTFLoc; //!< uniform location for time 
static GLint _dTimeLoc; //!< uniform location for delta time
static GLint _aLoc; //!< uniform location for acceleration

static GLint _texture0Loc; //!< uniform location for texture 0

// Attribute & uniform variable location indexes
static GLint _cPLoc;   //!< attribute location for vertex position
static GLint _cCLoc;   //!< attribute location for vertex color
static GLint _cGLoc;   //!< uniform location for gamma value
static GLint _cMvpLoc; //!< uniform location for modelview-projection matrix


//-----------------------------------------------------------------------------
void buildBox()
{
    // create C arrays on heap
    // Define the vertex position and colors as an array of structure
    // We define the colors with the same components as the cubes corners.
    _numV              = 8;
    VertexPC* vertices = new VertexPC[_numV];
    vertices[0].set(0.05, -0.5, 0.05, 1, 1, 1); //LTN
    vertices[1].set(0.05, -0.6, 0.05, 1, 0, 1);  //LBN
    vertices[2].set(0.05, -0.6, -0.05, 1, 0, 0); //LBF
    vertices[3].set(0.05, -0.5, -0.05, 1, 1, 0); //LTF
    vertices[4].set(-0.05, -0.6, -0.05, 0, 0, 0); //RBF
    vertices[5].set(-0.05, -0.6, 0.05, 0, 0, 1); //RBN
    vertices[6].set(-0.05, -0.5, 0.05, 0, 1, 1); //RTN
    vertices[7].set(-0.05, -0.5, -0.05, 0, 1, 0); //RTF

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
    glUtils::buildVAO(_vaoC,
                      _vboVC,
                      _vboIC,
                      vertices,
                      (GLint)_numV,
                      sizeof(VertexPC),
                      indices,
                      (GLint)_numI,
                      sizeof(GL_UNSIGNED_INT),
                      (GLint)_shaderProgID,
                      _cPLoc,
                      _cCLoc);

    // delete data on heap. The VBOs are now on the GPU
    delete[] vertices;
    delete[] indices;
}

float randomFloat(float a, float b)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff   = b - a;
    float r      = random * diff;
    return a + r;
}

//-----------------------------------------------------------------------------
void initParticles(float timeToLive, SLVec3f particleGenPos)
{
    _ttl            = timeToLive;
    Particle* data  = new Particle[AMOUNT];
    Particle p      = Particle();
    p.p             = particleGenPos;
    for (unsigned int i = 0; i < AMOUNT; i++)
    {
        p.v.x         = randomFloat(0.2f, -0.2f);   // Random value for x velocity
        p.v.y         = randomFloat(0.4f, 1.0f);    // Random value for y velocity
        p.v.z         = randomFloat(0.2f, -0.2f);   // Random value for z velocity
        p.initV       = p.v;                        // Initial velocity is set after the computation of the velocity
        p.st          = i * (timeToLive / AMOUNT);  // When the first particle dies the last one begin to live

        data[i] = p;
    }

    glGenTransformFeedbacks(2, _tfo);
    glGenVertexArrays(2, _vao);
    glGenBuffers(2, _vbo);

    for (unsigned int i = 0; i < 2; i++)
    {
        glBindVertexArray(_vao[i]);
        glBindBuffer(GL_ARRAY_BUFFER, _vbo[i]);
        glBufferData(GL_ARRAY_BUFFER, (AMOUNT * sizeof(Particle)), data, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(7 * sizeof(float)));
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
    _tFShaderVertID = glUtils::buildShader(_projectRoot + "/data/shaders/ParticleTF.vert", GL_VERTEX_SHADER);
    _tFShaderFragID = glUtils::buildShader(_projectRoot + "/data/shaders/ParticleTF.frag", GL_FRAGMENT_SHADER);
    _tFShaderProgID = glUtils::buildProgramTF(_tFShaderVertID, _tFShaderFragID);

    // Load, compile & link shaders
    _shaderVertID = glUtils::buildShader(_projectRoot + "/data/shaders/Particle.vert", GL_VERTEX_SHADER);
    _shaderFragID = glUtils::buildShader(_projectRoot + "/data/shaders/Particle.frag", GL_FRAGMENT_SHADER);
    _shaderGeomID = glUtils::buildShader(_projectRoot + "/data/shaders/Particle.geom", GL_GEOMETRY_SHADER);
    _shaderProgID = glUtils::buildProgram(_shaderVertID,_shaderGeomID, _shaderFragID);

    // Load, compile & link shaders
    _cShaderVertID = glUtils::buildShader(_projectRoot + "/data/shaders/ColorAttribute.vert", GL_VERTEX_SHADER);
    _cShaderFragID = glUtils::buildShader(_projectRoot + "/data/shaders/Color.frag", GL_FRAGMENT_SHADER);
    _cShaderProgID = glUtils::buildProgram(_cShaderVertID, _cShaderFragID);

    // Activate the shader program
    glUseProgram(_tFShaderProgID);

    // Get the variable locations (identifiers) within the program
    _tTLTFLoc   = glGetUniformLocation(_tFShaderProgID, "u_tTL");
    _timeTFLoc  = glGetUniformLocation(_tFShaderProgID, "u_time");
    _dTimeLoc = glGetUniformLocation(_tFShaderProgID, "u_deltaTime");
    _aLoc     = glGetUniformLocation(_tFShaderProgID, "u_acceleration");

    // Activate the shader program
    glUseProgram(_cShaderProgID);

    // Get the variable locations (identifiers) within the program
    _cPLoc   = glGetAttribLocation(_cShaderProgID, "a_position");
    _cCLoc   = glGetAttribLocation(_cShaderProgID, "a_color");
    _cGLoc   = glGetUniformLocation(_cShaderProgID, "u_oneOverGamma");
    _cMvpLoc = glGetUniformLocation(_cShaderProgID, "u_mvpMatrix");

    // Activate the shader program
    glUseProgram(_shaderProgID);

    // Get the variable locations (identifiers) within the program
    _gLoc   = glGetUniformLocation(_shaderProgID, "u_oneOverGamma");
    _mvLoc = glGetUniformLocation(_shaderProgID, "u_mvMatrix");
    _pMatLoc     = glGetUniformLocation(_shaderProgID, "u_pMatrix");
    _cLoc        = glGetUniformLocation(_shaderProgID, "u_color");
    _sLoc        = glGetUniformLocation(_shaderProgID, "u_scale");
    _radiusLoc   = glGetUniformLocation(_shaderProgID, "u_radius");
    _tTLLoc      = glGetUniformLocation(_shaderProgID, "u_tTL");
    _timeLoc     = glGetUniformLocation(_shaderProgID, "u_time");
    _texture0Loc = glGetUniformLocation(_shaderProgID, "u_matTextureDiffuse0");

    buildBox(); // Init the Cube
    //buildSquare();
    initParticles(5.0f, SLVec3f(0, 0, 0));

    glClearColor(0.0f, 0.0f, 0.0f, 1); // Set the background color
    //glEnable(GL_DEPTH_TEST);           // Enables depth test
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
    glDeleteShader(_shaderGeomID);
    glDeleteShader(_shaderFragID);
    glDeleteProgram(_shaderProgID);

    glDeleteShader(_cShaderVertID);
    glDeleteShader(_cShaderFragID);
    glDeleteProgram(_cShaderProgID);

    glDeleteShader(_tFShaderVertID);
    glDeleteShader(_tFShaderFragID);
    glDeleteProgram(_tFShaderProgID);

    // Delete arrays & buffers on GPU
    glDeleteVertexArrays(2, _vao);
    glDeleteBuffers(2, _vbo);
    glDeleteTransformFeedbacks(2, _tfo);
    glDeleteVertexArrays(1, &_vaoC);
    glDeleteBuffers(1, &_vboVC);
    glDeleteBuffers(1, &_vboIC);
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
    //_modelMatrix.translate(0.0f, 0.5f, 0.0f);

    //4) Build the combined modelview matrix
    SLMat4f mvp(_projectionMatrix);
    SLMat4f mv(_viewMatrix);
    mv.multiply(_modelMatrix);
    mvp.multiply(mv);

    _currentTime = glfwGetTime();
    std::cout << _currentTime << std::endl;
    float delatTime = _currentTime - _lastTime;
    _lastTime       = _currentTime;

    /////////// Update pass ////////////////
    // Activate the shader program
    glUseProgram(_tFShaderProgID);
    // Set the uniforms for transform feedback shader
    glUniform1f(_tTLTFLoc, _ttl);
    glUniform1f(_timeTFLoc, _currentTime);
    glUniform3f(_aLoc, 1.0f, 1.0f, 1.0f);
    glUniform1f(_dTimeLoc, delatTime);

    // Disable rendering
    glEnable(GL_RASTERIZER_DISCARD);
    // Bind the feedback object for the buffers to be drawn next
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,_tfo[_drawBuf]);
    // Draw points from input buffer with transform feedback
    glBeginTransformFeedback(GL_POINTS);
    glBindVertexArray(_vao[1 - _drawBuf]);
    glDrawArrays(GL_POINTS, 0, AMOUNT);
    glEndTransformFeedback();
    // Enable rendering
    glDisable(GL_RASTERIZER_DISCARD);
    //////////// Render pass ///////////////
    // Activate the shader program
    glUseProgram(_shaderProgID);
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
    glUniform1f(_radiusLoc, 0.05f);
    glUniform4f(_cLoc, 1.0f,1.0f,1.0f,1.0f);

     // Un-bind the feedback object.
     glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    
    glEnable(GL_BLEND); // Activate transparency (blending)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // use additive blending to give it a 'glow' effect

    // Draw the particles from the feedback buffer
    glBindVertexArray(_vao[_drawBuf]);
    glDrawArrays(GL_POINTS, 0, AMOUNT);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // don't forget to reset to default blending mode
    glDisable(GL_BLEND);                               // Disable transparency

    // Swap buffers
    _drawBuf = 1 - _drawBuf;

    //////////// End  ///////////////

    glUseProgram(_cShaderProgID);
    glUniformMatrix4fv(_cMvpLoc, 1, 0, (float*)&mvp);
    glUniform1f(_gLoc, 1.0f);

    // Activate the vertex array
    glBindVertexArray(_vaoC);

    // Activate the index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _vboIC);

    // Draw cube with triangles by indexes
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
