//#############################################################################
//  File:      TextureMapping.cpp
//  Purpose:   Minimal core profile OpenGL application for ambient-diffuse-
//             specular lighting shaders with Textures.
//  Date:      February 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <Utils.h>
#include <GL/gl3w.h>    // OpenGL headers
#include <GLFW/glfw3.h> // GLFW GUI library
#include <SL.h>         // Basic SL type definitions
#include <CVImage.h>    // Image class for image loading
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
static SLfloat     _scr2fbX;     //!< Factor from screen to framebuffer coords
static SLfloat     _scr2fbY;     //!< Factor from screen to framebuffer coords

// GLobal application variables
static SLMat4f _modelMatrix;      //!< 4x4 view matrix
static SLMat4f _viewMatrix;       //!< 4x4 model matrix
static SLMat4f _projectionMatrix; //!< 4x4 projection matrix

static GLuint _vao  = 0; //!< ID of the vertex array object
static GLuint _vboV = 0; //!< ID of the VBO for vertex attributes
static GLuint _vboI = 0; //!< ID of the VBO for vertex index array

static GLuint _numV = 0; //!< NO. of vertices
static GLuint _numI = 0; //!< NO. of vertex indexes for triangles

static GLint _resolution; //!< resolution of sphere stack & slices

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

static SLVec4f _globalAmbi;    //!< global ambient intensity
static SLVec3f _lightPos;      //!< Light position in world space
static SLVec3f _lightDir;      //!< Light direction in world space
static SLVec4f _lightAmbient;  //!< Light ambient intensity
static SLVec4f _lightDiffuse;  //!< Light diffuse intensity
static SLVec4f _lightSpecular; //!< Light specular intensity
static SLVec4f _matAmbient;    //!< Material ambient reflection coeff.
static SLVec4f _matDiffuse;    //!< Material diffuse reflection coeff.
static SLVec4f _matSpecular;   //!< Material specular reflection coeff.
static SLVec4f _matEmissive;   //!< Material emissive coeff.
static float   _matShininess;  //!< Material shininess exponent

static GLuint _shaderVertID = 0; //!< vertex shader id
static GLuint _shaderFragID = 0; //!< fragment shader id
static GLuint _shaderProgID = 0; //!< shader program id
static GLuint _textureID    = 0; //!< texture id

static GLint _pLoc;              //!< attribute location for vertex position
static GLint _nLoc;              //!< attribute location for vertex normal
static GLint _tLoc;              //!< attribute location for vertex texcoords
static GLint _mvpMatrixLoc;      //!< uniform location for modelview-projection matrix
static GLint _mvMatrixLoc;       //!< uniform location for modelview matrix
static GLint _nMatrixLoc;        //!< uniform location for normal matrix
static GLint _globalAmbiLoc;     //!< uniform location for global ambient intensity
static GLint _lightPosVSLoc;     //!< uniform location for light position in VS
static GLint _lightSpotDirVSLoc; //!< uniform location for light direction in VS
static GLint _lightAmbientLoc;   //!< uniform location for ambient light intensity
static GLint _lightDiffuseLoc;   //!< uniform location for diffuse light intensity
static GLint _lightSpecularLoc;  //!< uniform location for specular light intensity
static GLint _matAmbientLoc;     //!< uniform location for ambient light reflection
static GLint _matDiffuseLoc;     //!< uniform location for diffuse light reflection
static GLint _matSpecularLoc;    //!< uniform location for specular light reflection
static GLint _matEmissiveLoc;    //!< uniform location for light emission
static GLint _matShininessLoc;   //!< uniform location for shininess
static GLint _gLoc;              //!< uniform location for gamma value

static GLint _texture0Loc; //!< uniform location for texture 0

static const SLfloat PI = 3.14159265358979f;

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
    _numV               = (stacks + 1) * (slices + 1);
    VertexPNT* vertices = new VertexPNT[_numV];

    float  theta, dtheta; // angles around x-axis
    float  phi, dphi;     // angles around z-axis
    GLuint i, j;          // loop counters
    GLuint iv = 0;

    // init start values
    theta  = 0.0f;
    dtheta = Utils::PI / stacks;
    dphi   = 2.0f * Utils::PI / slices;

    // Define vertex position & normals by looping through all stacks
    for (i = 0; i <= stacks; ++i)
    {
        float sin_theta = sin(theta);
        float cos_theta = cos(theta);
        phi             = 0.0f;

        // Loop through all slices
        for (j = 0; j <= slices; ++j)
        {
            if (j == slices) phi = 0.0f;

            // define first the normal with length 1
            vertices[iv].n.x = sin_theta * cos(phi);
            vertices[iv].n.y = sin_theta * sin(phi);
            vertices[iv].n.z = cos_theta;

            // set the vertex position w. the scaled normal
            vertices[iv].p.x = radius * vertices[iv].n.x;
            vertices[iv].p.y = radius * vertices[iv].n.y;
            vertices[iv].p.z = radius * vertices[iv].n.z;

            // set the texture coords.
            vertices[iv].t.x = 0; // ???
            vertices[iv].t.y = 0; // ???

            phi += dphi;
            iv++;
        }
        theta += dtheta;
    }

    // create Index array x
    _numI           = (GLuint)(slices * stacks * 2 * 3);
    GLuint* indices = new GLuint[_numI];
    GLuint  ii      = 0, iV1, iV2;

    for (i = 0; i < stacks; ++i)
    {
        // index of 1st & 2nd vertex of stack
        iV1 = i * (slices + 1);
        iV2 = iV1 + slices + 1;

        for (j = 0; j < slices; ++j)
        { // 1st triangle ccw
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
                      vertices,
                      (GLint)_numV,
                      sizeof(VertexPNT),
                      indices,
                      (GLint)_numI,
                      sizeof(GL_UNSIGNED_INT),
                      (GLint)_shaderProgID,
                      _pLoc,
                      -1,
                      _nLoc);

    // Delete arrays on heap
    delete[] vertices;
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
                      _tLoc);

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
    _lightPos.set(0.0f, 0.0f, 100.0f);
    _lightDir.set(0.0f, 0.0f, -1.0f);
    _lightAmbient.set(0.1f, 0.1f, 0.1f);
    _lightDiffuse.set(1.0f, 1.0f, 1.0f);
    _lightSpecular.set(1.0f, 1.0f, 1.0f);
    _matAmbient.set(1.0f, 1.0f, 1.0f);
    _matDiffuse.set(1.0f, 1.0f, 1.0f);
    _matSpecular.set(1.0f, 1.0f, 1.0f);
    _matEmissive.set(0.0f, 0.0f, 0.0f);
    _matShininess = 100.0f;

    // backwards movement of the camera
    _camZ = -3.0f;

    // Mouse rotation parameters
    _rotX          = 0;
    _rotY          = 0;
    _deltaX        = 0;
    _deltaY        = 0;
    _mouseLeftDown = false;

    // Load textures
    _textureID = glUtils::buildTexture(_projectRoot + "/data/images/textures/earth1024_C.jpg");

    // Load, compile & link shaders
    _shaderVertID = glUtils::buildShader(_projectRoot + "/data/shaders/ADSTex.vert", GL_VERTEX_SHADER);
    _shaderFragID = glUtils::buildShader(_projectRoot + "/data/shaders/ADSTex.frag", GL_FRAGMENT_SHADER);
    _shaderProgID = glUtils::buildProgram(_shaderVertID, _shaderFragID);

    // Activate the shader program
    glUseProgram(_shaderProgID);

    // Get the variable locations (identifiers) within the vertex & pixel shader programs
    _pLoc              = glGetAttribLocation(_shaderProgID, "a_position");
    _nLoc              = glGetAttribLocation(_shaderProgID, "a_normal");
    _tLoc              = glGetAttribLocation(_shaderProgID, "a_texCoord");
    _mvMatrixLoc       = glGetUniformLocation(_shaderProgID, "u_mvMatrix");
    _mvpMatrixLoc      = glGetUniformLocation(_shaderProgID, "u_mvpMatrix");
    _nMatrixLoc        = glGetUniformLocation(_shaderProgID, "u_nMatrix");
    _globalAmbiLoc     = glGetUniformLocation(_shaderProgID, "u_globalAmbi");
    _lightPosVSLoc     = glGetUniformLocation(_shaderProgID, "u_lightPosVS");
    _lightSpotDirVSLoc = glGetUniformLocation(_shaderProgID, "u_lightSpotDirVS");
    _lightAmbientLoc   = glGetUniformLocation(_shaderProgID, "u_lightAmbient");
    _lightDiffuseLoc   = glGetUniformLocation(_shaderProgID, "u_lightDiffuse");
    _lightSpecularLoc  = glGetUniformLocation(_shaderProgID, "u_lightSpecular");
    _matAmbientLoc     = glGetUniformLocation(_shaderProgID, "u_matAmbient");
    _matDiffuseLoc     = glGetUniformLocation(_shaderProgID, "u_matDiffuse");
    _matSpecularLoc    = glGetUniformLocation(_shaderProgID, "u_matSpecular");
    _matEmissiveLoc    = glGetUniformLocation(_shaderProgID, "u_matEmissive");
    _matShininessLoc   = glGetUniformLocation(_shaderProgID, "u_matShininess");
    _texture0Loc       = glGetUniformLocation(_shaderProgID, "u_texture0");
    _gLoc              = glGetUniformLocation(_shaderProgID, "u_oneOverGamma");

    // Build object
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
void onClose(GLFWwindow* window)
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
    // Clear the color & depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // View transform: move the coordinate system away from the camera
    _viewMatrix.identity();
    _viewMatrix.translate(0, 0, _camZ);

    // View transform: rotate the coordinate system increasingly by the mouse
    _viewMatrix.rotate(_rotX + _deltaX, 1, 0, 0);
    _viewMatrix.rotate(_rotY + _deltaY, 0, 1, 0);

    // Transform light position & direction into view space
    SLVec3f lightPosVS = _viewMatrix * _lightPos;

    // The light dir is not a position. We only take the rotation of the mv matrix.
    SLMat3f viewRot    = _viewMatrix.mat3();
    SLVec3f lightDirVS = viewRot * _lightDir;

    // Rotate the model so that we see the square from the front side
    // or the earth from the equator.
    _modelMatrix.identity();
    _modelMatrix.rotate(90, -1, 0, 0);

    // Build the combined model-view and model-view-projection matrix
    SLMat4f mvp(_projectionMatrix);
    SLMat4f mv(_viewMatrix * _modelMatrix);
    mvp.multiply(mv);

    // Build normal matrix
    SLMat3f nm(mv.inverseTransposed());

    // Pass the matrix uniform variables
    glUniformMatrix4fv(_mvMatrixLoc, 1, 0, (float*)&mv);
    glUniformMatrix3fv(_nMatrixLoc, 1, 0, (float*)&nm);
    glUniformMatrix4fv(_mvpMatrixLoc, 1, 0, (float*)&mvp);

    // Pass lighting uniforms variables
    glUniform4fv(_globalAmbiLoc, 1, (float*)&_globalAmbi);
    glUniform3fv(_lightPosVSLoc, 1, (float*)&lightPosVS);
    glUniform3fv(_lightSpotDirVSLoc, 1, (float*)&lightDirVS);
    glUniform4fv(_lightAmbientLoc, 1, (float*)&_lightAmbient);
    glUniform4fv(_lightDiffuseLoc, 1, (float*)&_lightDiffuse);
    glUniform4fv(_lightSpecularLoc, 1, (float*)&_lightSpecular);
    glUniform4fv(_matAmbientLoc, 1, (float*)&_matAmbient);
    glUniform4fv(_matDiffuseLoc, 1, (float*)&_matDiffuse);
    glUniform4fv(_matSpecularLoc, 1, (float*)&_matSpecular);
    glUniform4fv(_matEmissiveLoc, 1, (float*)&_matEmissive);
    glUniform1f(_matShininessLoc, _matShininess);
    glUniform1i(_texture0Loc, 0);
    glUniform1f(_gLoc, 1.0f);

    //////////////////////
    // Draw with 2 VBOs //
    //////////////////////

    // Enable all of the vertex attribute arrays
    glEnableVertexAttribArray((GLuint)_pLoc);
    glEnableVertexAttribArray((GLuint)_nLoc);
    glEnableVertexAttribArray((GLuint)_tLoc);

    // Activate VBOs
    glBindBuffer(GL_ARRAY_BUFFER, _vboV);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _vboI);

    // Activate Texture
    glBindTexture(GL_TEXTURE_2D, _textureID);

    // For VBO only offset instead of data pointer
    GLsizei stride  = sizeof(VertexPNT);
    GLsizei offsetN = sizeof(SLVec3f);
    GLsizei offsetT = sizeof(SLVec3f) + sizeof(SLVec3f);
    glVertexAttribPointer((GLuint)_pLoc,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          stride,
                          nullptr);
    glVertexAttribPointer((GLuint)_nLoc,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          stride,
                          (void*)(size_t)offsetN);
    glVertexAttribPointer((GLuint)_tLoc,
                          2,
                          GL_FLOAT,
                          GL_FALSE,
                          stride,
                          (void*)(size_t)offsetT);

    ////////////////////////////////////////////////////////
    // Draw cube model triangles by indexes
    glDrawElements(GL_TRIANGLES,
                   (GLsizei)_numI,
                   GL_UNSIGNED_INT,
                   nullptr);
    ////////////////////////////////////////////////////////

    // Deactivate buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // Disable the vertex arrays
    glDisableVertexAttribArray((GLuint)_pLoc);
    glDisableVertexAttribArray((GLuint)_nLoc);
    glDisableVertexAttribArray((GLuint)_tLoc);

    // Check for errors from time to time
    GETGLERROR;

    // Fast copy the back buffer to the front buffer. This is OS dependent.
    glfwSwapBuffers(window);

    // Calculate frames per second
    char         title[255];
    static float lastTimeSec = 0;
    float        timeNowSec  = (float)glfwGetTime();
    float        fps         = calcFPS(timeNowSec - lastTimeSec);
    sprintf(title, "Sphere, %d x %d, fps: %4.0f", _resolution, _resolution, fps);
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
void onResize(GLFWwindow* window, int width, int height)
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
Mouse wheel eventhandler that moves the camera foreward or backwards
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
/*!
The C main procedure running the GLFW GUI application.
*/
int main(int argc, char* argv[])
{
    _projectRoot = SLstring(SL_PROJECT_ROOT);

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
#else
    //glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    //glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    //glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    _scrWidth  = 640;
    _scrHeight = 480;

    window = glfwCreateWindow(_scrWidth, _scrHeight, "My Title", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Get the current GL context. After this you can call GL
    glfwMakeContextCurrent(window);

    // On some systems screen & framebuffer size are different
    // All commands in GLFW are in screen coords but rendering in GL is
    // in framebuffer coords
    SLint fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    _scr2fbX = (float)fbWidth / (float)_scrWidth;
    _scr2fbY = (float)fbHeight / (float)_scrHeight;

    // Init OpenGL access library gl3w
    if (gl3wInit()!=0)
    {
        cerr << "Failed to initialize OpenGL" << endl;
        exit(-1);
    }

    // Check errors before we start
    GETGLERROR;

    glUtils::printGLInfo();

    glfwSetWindowTitle(window, "SLProject Test Application");

    // Set number of monitor refreshes between 2 buffer swaps
    glfwSwapInterval(1);

    onInit();
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
