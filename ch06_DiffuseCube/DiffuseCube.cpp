//#############################################################################
//  File:      DiffuseCube.cpp
//  Purpose:   Core profile OpenGL application with diffuse lighted cube with
//             GLFW as the OS GUI interface (http://www.glfw.org/).
//  Author:    Marcus Hudritsch
//  Date:      September 2012 (HS12)
//  Copyright: Marcus Hudritsch, Switzerland
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#include "stdafx.h"
#include "glUtils.h"   // Basics for OpenGL shaders, buffers & textures
#include "SL.h"        // Basic SL type definitions
#include "SLImage.h"   // Image class for image loading
#include "SLVec3.h"    // 3D vector class
#include "SLMat4.h"    // 4x4 matrix class
#include "../lib-SLExternal/glew/include/GL/glew.h"     // OpenGL headers
#include "../lib-SLExternal/glfw3/include/GLFW/glfw3.h" // GLFW GUI library

//-----------------------------------------------------------------------------
//! Struct definition for vertex attributes
struct VertexPN
{
   SLVec3f p;  // vertex position [x,y,z]
   SLVec3f n;  // vertex normal [x,y,z]

   // Setter method
   void set(float posX, float posY, float posZ,
            float normalX, float normalY, float normalZ)
   {  p.set(posX, posY, posZ);
      n.set(normalX, normalY, normalZ);
   }
};
//-----------------------------------------------------------------------------
// GLobal application variables
GLFWwindow* window;                 //!< The global glfw window handle
SLint     _scrWidth;                //!< Window width at start up
SLint     _scrHeight;               //!< Window height at start up
SLfloat   _scr2fbX;                 //!< Factor from screen to framebuffer coords
SLfloat   _scr2fbY;                 //!< Factor from screen to framebuffer coords

SLMat4f  _viewMatrix;               //!< 4x4 view matrix
SLMat4f  _modelMatrix;              //!< 4x4 model matrix
SLMat4f  _projectionMatrix;         //!< 4x4 projection matrix

VertexPN* _v=0;                     //!< Array of vertices
SLVec3f*  _p=0;                     //!< Array for vertex positions
SLVec3f*  _n=0;                     //!< Array for vertex normals
GLubyte*  _i=0;                     //!< 36 indexes for 2 triangles per cube side
GLuint   _vboV = 0;                 //!< ID of the VBO for vertex array
GLuint   _vboI = 0;                 //!< ID of the VBO for vertex index array

float    _camZ;                     //!< z-distance of camera
float    _rotX, _rotY;              //!< rotation angles around x & y axis
int      _deltaX, _deltaY;          //!< delta mouse motion
int      _startX, _startY;          //!< x,y mouse start positions
int      _mouseX, _mouseY;          //!< current mouse position
bool     _mouseLeftDown;            //!< Flag if mouse is down
GLuint   _modifiers = 0;            //!< modifier bit flags
const GLuint NONE  = 0;             //!< constant for no modifier
const GLuint SHIFT = 0x00200000;    //!< constant for shift key modifier
const GLuint CTRL  = 0x00400000;    //!< constant for control key modifier
const GLuint ALT   = 0x00800000;    //!< constant for alt key modifier

GLuint   _shaderVertID = 0;         //! vertex shader id
GLuint   _shaderFragID = 0;         //! fragment shader id
GLuint   _shaderProgID = 0;         //! shader program id

// Attribute & uniform variable location indexes
GLint    _pLoc;            //!< attribute location for vertex position
GLint    _nLoc;            //!< attribute location for vertex normal
GLint    _mvpLoc;          //!< uniform location for modelview-projection matrix
GLint    _mvLoc;           //!< uniform location for modelview matrix
GLint    _nmLoc;           //!< uniform location for normal matrix
GLint    _lightDirVSLoc;   //!< uniform location for light direction in VS
GLint    _lightDiffuseLoc; //!< uniform location for diffuse light intensity
GLint    _matDiffuseLoc;   //!< uniform location for diffuse light refelction

//-----------------------------------------------------------------------------
void buildBox()
{
    // create arrays
    _p = new SLVec3f[24];
    _n = new SLVec3f[24];
    _v = new VertexPN[24];
    _i = new GLubyte[36];

    // Define the vertex pos. and normals as a structure of arrays (_p & _n)
    // Define the vertex pos. and normals as an array of structure (_v)
    _p[ 0].set(1, 1, 1); _n[ 0].set( 1, 0, 0); _v[ 0].set(1, 1, 1,  1, 0, 0);
    _p[ 1].set(1, 0, 1); _n[ 1].set( 1, 0, 0); _v[ 1].set(1, 0, 1,  1, 0, 0);
    _p[ 2].set(1, 0, 0); _n[ 2].set( 1, 0, 0); _v[ 2].set(1, 0, 0,  1, 0, 0);
    _p[ 3].set(1, 1, 0); _n[ 3].set( 1, 0, 0); _v[ 3].set(1, 1, 0,  1, 0, 0);
    _p[ 4].set(1, 1, 0); _n[ 4].set( 0, 0,-1); _v[ 4].set(1, 1, 0,  0, 0,-1);
    _p[ 5].set(1, 0, 0); _n[ 5].set( 0, 0,-1); _v[ 5].set(1, 0, 0,  0, 0,-1);
    _p[ 6].set(0, 0, 0); _n[ 6].set( 0, 0,-1); _v[ 6].set(0, 0, 0,  0, 0,-1);
    _p[ 7].set(0, 1, 0); _n[ 7].set( 0, 0,-1); _v[ 7].set(0, 1, 0,  0, 0,-1);
    _p[ 8].set(0, 0, 1); _n[ 8].set(-1, 0, 0); _v[ 8].set(0, 0, 1, -1, 0, 0);
    _p[ 9].set(0, 1, 1); _n[ 9].set(-1, 0, 0); _v[ 9].set(0, 1, 1, -1, 0, 0);
    _p[10].set(0, 1, 0); _n[10].set(-1, 0, 0); _v[10].set(0, 1, 0, -1, 0, 0);
    _p[11].set(0, 0, 0); _n[11].set(-1, 0, 0); _v[11].set(0, 0, 0, -1, 0, 0);
    _p[12].set(1, 1, 1); _n[12].set( 0, 0, 1); _v[12].set(1, 1, 1,  0, 0, 1);
    _p[13].set(0, 1, 1); _n[13].set( 0, 0, 1); _v[13].set(0, 1, 1,  0, 0, 1);
    _p[14].set(0, 0, 1); _n[14].set( 0, 0, 1); _v[14].set(0, 0, 1,  0, 0, 1);
    _p[15].set(1, 0, 1); _n[15].set( 0, 0, 1); _v[15].set(1, 0, 1,  0, 0, 1);
    _p[16].set(1, 1, 1); _n[16].set( 0, 1, 0); _v[16].set(1, 1, 1,  0, 1, 0);
    _p[17].set(1, 1, 0); _n[17].set( 0, 1, 0); _v[17].set(1, 1, 0,  0, 1, 0);
    _p[18].set(0, 1, 0); _n[18].set( 0, 1, 0); _v[18].set(0, 1, 0,  0, 1, 0);
    _p[19].set(0, 1, 1); _n[19].set( 0, 1, 0); _v[19].set(0, 1, 1,  0, 1, 0);
    _p[20].set(0, 0, 0); _n[20].set( 0,-1, 0); _v[20].set(0, 0, 0,  0,-1, 0);
    _p[21].set(1, 0, 0); _n[21].set( 0,-1, 0); _v[21].set(1, 0, 0,  0,-1, 0);
    _p[22].set(1, 0, 1); _n[22].set( 0,-1, 0); _v[22].set(1, 0, 1,  0,-1, 0);
    _p[23].set(0, 0, 1); _n[23].set( 0,-1, 0); _v[23].set(0, 0, 1,  0,-1, 0);

    // Define the triangle indexes of the cubes vertices
    int n = 0;
    _i[n++] =  0; _i[n++] =  1; _i[n++] =  2;  _i[n++] =  0; _i[n++] =  2; _i[n++] =  3;
    _i[n++] =  4; _i[n++] =  5; _i[n++] =  6;  _i[n++] =  4; _i[n++] =  6; _i[n++] =  7;
    _i[n++] =  8; _i[n++] =  9; _i[n++] = 10;  _i[n++] =  8; _i[n++] = 10; _i[n++] = 11;
    _i[n++] = 12; _i[n++] = 13; _i[n++] = 14;  _i[n++] = 12; _i[n++] = 14; _i[n++] = 15;
    _i[n++] = 16; _i[n++] = 17; _i[n++] = 18;  _i[n++] = 16; _i[n++] = 18; _i[n++] = 19;
    _i[n++] = 20; _i[n++] = 21; _i[n++] = 22;  _i[n++] = 20; _i[n++] = 22; _i[n++] = 23;

    // Create vertex buffer objects
    _vboV = glUtils::buildVBO(_v, 24, 6, sizeof(GLfloat), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    _vboI = glUtils::buildVBO(_i, 36, 1, sizeof(GLubyte), GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
}
//-----------------------------------------------------------------------------
/*!
onInit initializes the global variables and builds the shader program. It
should be called after a window with a valid OpenGL context is present.
*/
void onInit()
{
   buildBox();

   // backwards movement of the camera
   _camZ = -4;

   // Mouse rotation parameters
   _rotX = _rotY = 0;
   _deltaX = _deltaY = 0;
   _mouseLeftDown = false;

   // Load, compile & link shaders
   _shaderVertID = glUtils::buildShader("../_data/shaders/Diffuse.vert", GL_VERTEX_SHADER);
   _shaderFragID = glUtils::buildShader("../_data/shaders/Diffuse.frag", GL_FRAGMENT_SHADER);
   _shaderProgID = glUtils::buildProgram(_shaderVertID, _shaderFragID);

   // Activate the shader program
   glUseProgram(_shaderProgID);

   // Get the variable locations (identifiers) within the program
   _pLoc            = glGetAttribLocation (_shaderProgID, "a_position");
   _nLoc            = glGetAttribLocation (_shaderProgID, "a_normal");
   _mvpLoc          = glGetUniformLocation(_shaderProgID, "u_mvpMatrix");
   _nmLoc           = glGetUniformLocation(_shaderProgID, "u_nMatrix");
   _lightDirVSLoc   = glGetUniformLocation(_shaderProgID, "u_lightDirVS");
   _lightDiffuseLoc = glGetUniformLocation(_shaderProgID, "u_lightDiffuse");
   _matDiffuseLoc   = glGetUniformLocation(_shaderProgID, "u_matDiffuse");

   glClearColor(0.5f, 0.5f, 0.5f, 1);  // Set the background color
   glEnable(GL_DEPTH_TEST);            // Enables depth test
   glEnable(GL_CULL_FACE);             // Enables the culling of back faces
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
   _viewMatrix.rotate(_rotX + _deltaX, 1,0,0);
   _viewMatrix.rotate(_rotY + _deltaY, 0,1,0);

   // Model transform: move the cube so that it rotates around its center
   _modelMatrix.identity();
   _modelMatrix.translate(-2.0f, -0.5f, -0.5f);

   // Build the combined modelview-projection matrix
   SLMat4f mvp(_projectionMatrix);
   SLMat4f mv(_viewMatrix);
   mv.multiply(_modelMatrix);
   mvp.multiply(mv);

   // Build normal matrix
   SLMat3f nm(mv.inverseTransposed());

   // Pass the uniform variables to the shader
   glUniformMatrix4fv(_mvpLoc, 1, 0, (float*)&mvp);
   glUniformMatrix3fv(_nmLoc,   1, 0, (float*)&nm);
   glUniform3f(_lightDirVSLoc, 0.5f, 1.0f, 1.0f);          // light direction in view space
   glUniform4f(_lightDiffuseLoc, 1.0f, 1.0f,  1.0f, 1.0f);  // diffuse light intensity (RGBA)
   glUniform4f(_matDiffuseLoc, 1.0f, 0.0f, 0.0f, 1.0f);     // diffuse material reflection (RGBA)

   GETGLERROR;


   /////////////////////////////////
   // Draw red cube with 3 arrays //
   /////////////////////////////////

   // Enable the attributes as arrays (not constant)
   glEnableVertexAttribArray(_pLoc);
   glEnableVertexAttribArray(_nLoc);

   // Set the vertex attribute pointers to our vertex arrays
   glVertexAttribPointer(_pLoc, 3, GL_FLOAT, GL_FALSE, 0, _p);
   glVertexAttribPointer(_nLoc, 3, GL_FLOAT, GL_FALSE, 0, _n);

   // Draw cube with triangles by indexes
   glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_BYTE, _i);

   ///////////////////////////////////
   // Draw green cube with 2 arrays //
   ///////////////////////////////////

   // Transform 2 units to the left & rebuild mvp
   _modelMatrix.translate(1.5f, 0.0f, 0.0f);
   mvp.setMatrix(_projectionMatrix * _viewMatrix * _modelMatrix);

   // Pass updated mvp and set the red color
   glUniformMatrix4fv(_mvpLoc, 1, 0, (float*)&mvp);
   glUniform4f(_matDiffuseLoc, 0.0f, 1.0f, 0.0f, 1.0f);

   // Set the vertex attribute pointers to the array of structs
   GLsizei stride = sizeof(VertexPN);
   glVertexAttribPointer(_pLoc, 3, GL_FLOAT, GL_FALSE, stride, &_v[0].p.x);
   glVertexAttribPointer(_nLoc, 3, GL_FLOAT, GL_FALSE, stride, &_v[0].n.x);

   // Draw cube with triangles by indexes
   glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_BYTE, _i);

   ////////////////////////////////
   // Draw blue cube with 2 VBOs //
   ////////////////////////////////

   // Transform 2 units to the left & rebuild mvp
   _modelMatrix.translate(1.5f, 0.0f, 0.0f);
   mvp.setMatrix(_projectionMatrix * _viewMatrix * _modelMatrix);

   // Pass updated mvp and set the blue color
   glUniformMatrix4fv(_mvpLoc, 1, 0, (float*)&mvp);
   glUniform4f(_matDiffuseLoc, 0.0f, 0.0f, 1.0f, 1.0f);

   // Activate VBOs
   glBindBuffer(GL_ARRAY_BUFFER, _vboV);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _vboI);

   // For VBO only offset instead of data pointer
   GLsizei offsetN = sizeof(SLVec3f);
   glVertexAttribPointer(_pLoc, 3, GL_FLOAT, GL_FALSE, stride, 0);
   glVertexAttribPointer(_nLoc, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetN);

   // Draw cube with triangles by indexes
   glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_BYTE, 0);

   // Deactivate buffers
   glBindBuffer(GL_ARRAY_BUFFER, 0);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

   // Disable the vertex arrays
   glDisableVertexAttribArray(_pLoc);
   glDisableVertexAttribArray(_nLoc);

   // Fast copy the back buffer to the front buffer. This is OS dependent.
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
   double w = (double)width;
   double h = (double)height;

   // define the projection matrix
   _projectionMatrix.perspective(50, w/h, 0.01f, 10.0f);

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

    _mouseLeftDown = (action==GLFW_PRESS);
    if (_mouseLeftDown)
    {   _startX = x;
        _startY = y;

        // Renders only the lines of a polygon during mouse moves
        if (button==GLFW_MOUSE_BUTTON_RIGHT)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else
    {   _rotX += _deltaX;
        _rotY += _deltaY;
        _deltaX = 0;
        _deltaY = 0;

        // Renders filled polygons
        if (button==GLFW_MOUSE_BUTTON_RIGHT)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
}
//-----------------------------------------------------------------------------
/*!
Mouse move eventhandler tracks the mouse delta since touch down (_deltaX/_deltaY)
*/
void onMouseMove(GLFWwindow* window, double x, double y)
{
    _mouseX  = (int)x;
    _mouseY  = (int)y;

    if (_mouseLeftDown)
    {   _deltaY = (int)x - _startX;
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
        _camZ += (SLfloat)SL_sign(yscroll)*0.1f;
        onPaint();
    }
}
//-----------------------------------------------------------------------------
/*!
Key action eventhandler handles key down & release events
*/
void onKey(GLFWwindow* window, int GLFWKey, int scancode, int action, int mods)
{
    if (action==GLFW_PRESS)
    {
        switch (GLFWKey)
        {
            case GLFW_KEY_ESCAPE:
                onClose(window);
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_LEFT_SHIFT:     _modifiers = _modifiers|SHIFT; break;
            case GLFW_KEY_RIGHT_SHIFT:    _modifiers = _modifiers|SHIFT; break;
            case GLFW_KEY_LEFT_CONTROL:   _modifiers = _modifiers|CTRL; break;
            case GLFW_KEY_RIGHT_CONTROL:  _modifiers = _modifiers|CTRL; break;
            case GLFW_KEY_LEFT_ALT:       _modifiers = _modifiers|ALT; break;
            case GLFW_KEY_RIGHT_ALT:      _modifiers = _modifiers|ALT; break;
        }
    } else
    if (action == GLFW_RELEASE)
    {   switch (GLFWKey)
        {   case GLFW_KEY_LEFT_SHIFT:     _modifiers = _modifiers^SHIFT; break;
            case GLFW_KEY_RIGHT_SHIFT:    _modifiers = _modifiers^SHIFT; break;
            case GLFW_KEY_LEFT_CONTROL:   _modifiers = _modifiers^CTRL; break;
            case GLFW_KEY_RIGHT_CONTROL:  _modifiers = _modifiers^CTRL; break;
            case GLFW_KEY_LEFT_ALT:       _modifiers = _modifiers^ALT; break;
            case GLFW_KEY_RIGHT_ALT:      _modifiers = _modifiers^ALT; break;
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
int main()
{
    if (!glfwInit())
    {   fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(onGLFWError);

    // Enable fullscreen anti aliasing with 4 samples
    glfwWindowHint(GLFW_SAMPLES, 4);

    _scrWidth = 640;
    _scrHeight = 480;

    window = glfwCreateWindow(_scrWidth, _scrHeight, "My Title", NULL, NULL);
    if (!window)
    {   glfwTerminate();
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

    // Include OpenGL via GLEW (init must be after window creation)
    // The goal of the OpenGL Extension Wrangler Library (GLEW) is to assist C/C++ 
    // OpenGL developers with two tedious tasks: initializing and using extensions 
    // and writing portable applications. GLEW provides an efficient run-time 
    // mechanism to determine whether a certain extension is supported by the 
    // driver or not. OpenGL core and extension functionality is exposed via a 
    // single header file. Download GLEW at: http://glew.sourceforge.net/
    glewExperimental = GL_TRUE;  // avoids a crash
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {   fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    glUtils::printGLInfo();

    glfwSetWindowTitle(window, "Diffuse Cubes");

    // Set number of monitor refreshes between 2 buffer swaps
    glfwSwapInterval(1);

    onInit();
    onResize(window, (SLint)(_scrWidth  * _scr2fbX),
                     (SLint)(_scrHeight * _scr2fbY));

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
      else glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(0);
}
//-----------------------------------------------------------------------------
