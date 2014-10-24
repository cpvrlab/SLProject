//#############################################################################
//  File:      TextureMapping.cpp
//  Purpose:   Core profile OpenGL application for ambient-diffuse-specular
//             lighting shaders with Textures.
//  Author:    Marcus Hudritsch
//  Date:      September 2011 (HS11)
//  Copyright: M. Hudritsch, Fachhochschule Nordwestschweiz
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT
#include <nvwa/debug_new.h>   // memory leak detector
#endif
#include <SLImage.h>          // Image loading classes

using namespace std;

//-----------------------------------------------------------------------------
//! Struct defintion for vertex with position & normal attribute
struct VertexPNT
{  SLVec3f p;
   SLVec3f n;
   SLVec2f t;
};
//-----------------------------------------------------------------------------
// GLobal application variables
SLMat4f  _modelViewMatrix;          //!< 4x4 modelview matrix
SLMat4f  _projectionMatrix;         //!< 4x4 projection matrix

GLuint   _numI = 0;                 //!< NO. of vertex indexes for triangles
GLuint   _vboV = 0;                 //!< ID of the VBO for vertex array
GLuint   _vboI = 0;                 //!< ID of the VBO for vertex index array
GLint    _resolution;               //!< resolution of sphere stack & slices

float    _camZ;                     //!< z-distance of camera
float    _rotX, _rotY;              //!< rotation angles around x & y axis
int      _deltaX, _deltaY;          //!< delta mouse motion
int      _startX, _startY;          //!< x,y mouse start positions
int      _mouseX, _mouseY;          //!< current mouse position
bool     _mouseDown;                //!< flag if mouse is down
GLuint   _modifiers = 0;            //!< modifier bit flags
const GLuint NONE  = 0;             //!< constant for no modifier
const GLuint SHIFT = 0x00200000;    //!< constant for shift key modifier
const GLuint CTRL  = 0x00400000;    //!< constant for control key modifier
const GLuint ALT   = 0x00800000;    //!< constant for alt key modifier

SLVec4f  _globalAmbi;               //!< global ambient intensity
SLVec3f  _lightPos;                 //!< Light position in world space
SLVec3f  _lightDir;                 //!< Light direction in world space
SLVec4f  _lightAmbi;                //!< Light ambient intensity   
SLVec4f  _lightDiff;                //!< Light diffuse intensity   
SLVec4f  _lightSpec;                //!< Light specular intensity   
SLVec3f  _lightAtt;                 //!< Light attenuation coeff. c1, c2 & c3
SLVec4f  _matAmbi;                  //!< Material ambient reflection coeff.
SLVec4f  _matDiff;                  //!< Material diffuse reflection coeff.
SLVec4f  _matSpec;                  //!< Material specular reflection coeff.
SLVec4f  _matEmis;                  //!< Material emissive coeff.
float    _matShine;                 //!< Material shininess

GLuint   _shaderVertID = 0;         //!< vertex shader id
GLuint   _shaderFragID = 0;         //!< fragment shader id
GLuint   _shaderProgID = 0;         //!< shader program id
GLuint   _textureID0 = 0;           //!< texture id
GLuint   _textureID1 = 0;           //!< texture id

// Attribute & uniform variable location indexes    
GLint    _pLoc;            //!< attribute location for vertex position
GLint    _nLoc;            //!< attribute location for vertex normal
GLint    _tLoc;            //!< attribute location for vertex texcoords
GLint    _mvpLoc;          //!< uniform location for modelview-projection matrix
GLint    _mvLoc;           //!< uniform location for modelview matrix
GLint    _nmLoc;           //!< uniform location for normal matrix

GLint    _globalAmbiLoc;   //!< uniform location for global ambient intensity
GLint    _lightPosLoc;     //!< uniform location for light position in VS 
GLint    _lightDirLoc;     //!< uniform location for light direction in VS 
GLint    _lightAmbiLoc;    //!< uniform location for ambient light intensity 
GLint    _lightDiffLoc;    //!< uniform location for diffuse light intensity 
GLint    _lightSpecLoc;    //!< uniform location for specular light intensity 
GLint    _lightSpotCutLoc; //!< uniform location for spot cutoff angle
GLint    _lightSpotCosLoc; //!< uniform location for cosine of spot cutoff angle
GLint    _lightSpotExpLoc; //!< uniform location for cosine of spot cutoff angle
GLint    _lightAttLoc;     //!< uniform location for light attenuation coeff.
GLint    _matAmbiLoc;      //!< uniform location for ambient light refelction
GLint    _matDiffLoc;      //!< uniform location for diffuse light refelction
GLint    _matSpecLoc;      //!< uniform location for specular light refelction
GLint    _matEmisLoc;      //!< uniform location for light emission
GLint    _matShineLoc;     //!< uniform location for shininess

GLint    _texture0Loc;     //!< uniform location for texture 0
GLint    _texture1Loc;     //!< uniform location for texture 1

static const SLfloat PI = 3.14159265358979f;

//-----------------------------------------------------------------------------
/*!
buildSphere creates the vertex attributes for a sphere and creates the VBO
at the end. The sphere is built in stacks & slices.
*/
void buildSphere(float radius, int stacks, int slices)
{  
   assert(stacks > 3 && slices > 3);

   // create vertex array
   GLuint numV = (stacks+1) * (slices+1);
   VertexPNT* v = new VertexPNT[numV];

   float  theta, dtheta; // angles around x-axis
   float  phi, dphi;     // angles around z-axis
   float  s, t, ds, dt;  // texture coords
   int    i, j;          // loop counters
   GLuint iv = 0;
   
   // init start values
   theta = 0.0f;
   dtheta = PI / stacks;
   dphi = 2.0f * PI / slices;
   ds = 1.0f / slices;
   dt = 1.0f / stacks;
   t = 1.0f;
   
   // Define vertex position & normals by looping through all stacks
   for (i=0; i<=stacks; ++i)
   {  
      float sin_theta  = sin(theta);
      float cos_theta  = cos(theta);
      phi = s = 0.0f;

      // Loop through all slices
      for (j = 0; j<=slices; ++j)
      {  
         if (j==slices) phi = 0.0f;

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
         s += ds;
         iv++;
      }
      theta += dtheta;
      t -= dt;
   }

   // create Index array x
   _numI = slices * stacks * 2 * 3;
   GLuint* x = new GLuint[_numI];
   GLuint ii = 0, iV1, iV2;

   for (i=0; i<stacks; ++i)
   {  
      // index of 1st & 2nd vertex of stack
      iV1 = i * (slices+1);
      iV2 = iV1 + slices + 1;

      for (j = 0; j<slices; ++j)
      {  // 1st triangle ccw         
         x[ii++] = iV1+j;
         x[ii++] = iV2+j;
         x[ii++] = iV2+j+1;
         // 2nd triangle ccw
         x[ii++] = iV1+j;
         x[ii++] = iV2+j+1;
         x[ii++] = iV1+j+1;
      }
   }

   // Create vertex buffer objects
   _vboV = glUtils::buildVBO(v,  numV, 8, sizeof(GLfloat), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
   _vboI = glUtils::buildVBO(x, _numI, 1, sizeof(GLuint), GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);

   // Delete arrays on heap
   delete[] v;
   delete[] x;
}
//-----------------------------------------------------------------------------
/*!
buildSquare creates the vertex attributes for a textured square and VBO.
*/
void buildSquare()
{
   // create vertex array for interleaved position, normal and texCoord
	//            Position,  Normal  , texCrd,
   float v[] = {-1, 0, -1,  0, -1, 0,  0,  0, // Vertex 0
                 1, 0, -1,  0, -1, 0,  1,  0, // Vertex 1
                 1, 0,  1,  0, -1, 0,  1,  1, // Vertex 2
                -1, 0,  1,  0, -1, 0,  0,  1};// Vertex 3
   _vboV = glUtils::buildVBO(v, 4, 8, sizeof(GLfloat), GL_ARRAY_BUFFER, GL_STATIC_DRAW);

   // create index array for GL_TRIANGLES
   _numI = 6;
   GLuint i[] = {0, 1, 2,  0, 2, 3};
   _vboI = glUtils::buildVBO(i, _numI, 1, sizeof(GLuint), GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
}
//-----------------------------------------------------------------------------
/*!
calcFPS determines the frame per second measurement by averaging 60 frames.
*/
float calcFPS(float deltaTime)
{  const  SLint   FILTERSIZE = 60;
   static SLfloat frameTimes[FILTERSIZE];
   static SLuint  frameNo = 0;

   frameTimes[frameNo % FILTERSIZE] = deltaTime;
   float sumTime = 0.0f;
   for (SLuint i=0; i<FILTERSIZE; ++i) sumTime += frameTimes[i];
   frameNo++;
   float frameTimeSec = sumTime / (SLfloat)FILTERSIZE;
   float fps = 1 / frameTimeSec;

   return fps;
}
//-----------------------------------------------------------------------------
/*!
onInit initializes the global variables and builds the shader program. It
should be called after a window with a valid OpenGL context is present.
*/
void onInit()
{  
   // Define sphere
   _resolution = 64;
   
   //buildSphere(1.0f, _resolution, _resolution);
   buildSquare();

   // Set light parameters
   _globalAmbi.set(0.0f, 0.0f, 0.0f);
   _lightPos.set( 0.0f, 0.0f, 100.0f);
   _lightDir.set( 0.0f, 0.0f,-1.0f);
   _lightAmbi.set( 0.1f, 0.1f, 0.1f);  
   _lightDiff.set( 1.0f, 1.0f, 1.0f);
   _lightSpec.set( 1.0f, 1.0f, 1.0f);
   _lightAtt.set( 1.0f, 0.0f, 0.0f);
   _matAmbi.set( 1.0f, 1.0f, 1.0f);    
   _matDiff.set( 1.0f, 1.0f, 1.0f);    
   _matSpec.set( 1.0f, 1.0f, 1.0f);    
   _matEmis.set( 0.0f, 0.0f, 0.0f);
   _matShine = 500.0f; 
   
   // backwards movement of the camera
   _camZ = -3.0f;      

   // Mouse rotation paramters
   _rotX = 0;
   _rotY = 0;
   _deltaX = _deltaY = 0;
   _mouseDown = false;

   // Build textures
   _textureID0 = glUtils::buildTexture("../_data/images/textures/earth2048_C.jpg");
   _textureID1 = glUtils::buildTexture("../_data/images/textures/earth2048_G.jpg");

   // Load, compile & link shaders
   _shaderVertID = glUtils::buildShader("../lib-SLProject/source/oglsl/ADSTex_Loesung.vert", GL_VERTEX_SHADER);
   _shaderFragID = glUtils::buildShader("../lib-SLProject/source/oglsl/ADSTex_Loesung.frag", GL_FRAGMENT_SHADER);
   _shaderProgID = glUtils::buildProgram(_shaderVertID, _shaderFragID);;

   // Activate the shader programm
   glUseProgram(_shaderProgID); 

   // Get the variable locations (identifiers) within the program
   _pLoc            = glGetAttribLocation (_shaderProgID, "a_position");
   _nLoc            = glGetAttribLocation (_shaderProgID, "a_normal");
   _tLoc            = glGetAttribLocation (_shaderProgID, "a_texCoord");

   _mvLoc           = glGetUniformLocation(_shaderProgID, "u_mvMatrix");
   _mvpLoc          = glGetUniformLocation(_shaderProgID, "u_mvpMatrix");
   _nmLoc           = glGetUniformLocation(_shaderProgID, "u_nMatrix");
   
   _globalAmbiLoc   = glGetUniformLocation(_shaderProgID, "u_globalAmbi");
   _lightPosLoc     = glGetUniformLocation(_shaderProgID, "u_lightPosVS");
   _lightDirLoc     = glGetUniformLocation(_shaderProgID, "u_lightDirVS");
   _lightAmbiLoc    = glGetUniformLocation(_shaderProgID, "u_lightAmbi");
   _lightDiffLoc    = glGetUniformLocation(_shaderProgID, "u_lightDiff");
   _lightSpecLoc    = glGetUniformLocation(_shaderProgID, "u_lightSpec");
   _lightAttLoc     = glGetUniformLocation(_shaderProgID, "u_lightAtt");
   _matAmbiLoc      = glGetUniformLocation(_shaderProgID, "u_matAmbi");
   _matDiffLoc      = glGetUniformLocation(_shaderProgID, "u_matDiff");
   _matSpecLoc      = glGetUniformLocation(_shaderProgID, "u_matSpec");
   _matEmisLoc      = glGetUniformLocation(_shaderProgID, "u_matEmis");
   _matShineLoc     = glGetUniformLocation(_shaderProgID, "u_matShine");

   _texture0Loc     = glGetUniformLocation(_shaderProgID, "u_texture0");
   _texture1Loc     = glGetUniformLocation(_shaderProgID, "u_texture1");      

   glClearColor(0.0f, 0.0f, 0.0f, 1);  // Set the background color         
   glEnable(GL_DEPTH_TEST);            // Enables depth test
   glEnable(GL_CULL_FACE);             // Enables the culling of back faces

   GET_GL_ERROR;                       // Check for OpenGL errors
}
//-----------------------------------------------------------------------------
/*!
onClose is called when the user closes the window and can be used for proper
deallocation of resources.
*/
int onClose()
{
   // Delete shaders & programs on GPU
   glDeleteShader(_shaderVertID);
   glDeleteShader(_shaderFragID);
   glDeleteProgram(_shaderProgID);
   
   // Delete arrays & buffers on GPU
   glDeleteBuffers(1, &_vboV);
   glDeleteBuffers(1, &_vboI);

   // Remove callback functions
   glfwSetWindowSizeCallback (NULL);
   glfwSetWindowCloseCallback(NULL);
   glfwSetMouseButtonCallback(NULL);
   glfwSetMousePosCallback   (NULL);
   glfwSetMouseWheelCallback (NULL);
   glfwSetWindowCloseCallback(NULL);

   // Return true for closing window
   return GL_TRUE;
}
//-----------------------------------------------------------------------------
/*!
onPaint does all the rendering for one frame from scratch with OpenGL.
*/
bool onPaint()
{  
   // Clear the color & depth buffer
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   // Start with identity every frame
   _modelViewMatrix.identity();
   
   // View transform: move the coordiante system away from the camera
   _modelViewMatrix.translate(0, 0, _camZ);

   // View transform: rotate the coordinate system increasingly
   _modelViewMatrix.rotate(_rotX + _deltaX, 1,0,0);
   _modelViewMatrix.rotate(_rotY + _deltaY, 0,1,0);

   // Transform light position & direction into view space
   SLVec3f lightPosVS = _modelViewMatrix * _lightPos;
   // The light direction is not a position. We therefore only take
   // the rotation part of the mv matrix.
   SLMat3f viewRot    = _modelViewMatrix.mat3();
   SLVec3f lightDirVS = viewRot * _lightDir;

   // Rotate the model 
   _modelViewMatrix.rotate(90, -1,0,0);

   // Build the combined modelview-projection matrix
   SLMat4f mvp(_projectionMatrix * _modelViewMatrix);

   // Pass the matrix uniform variables
   glUniformMatrix4fv(_mvLoc,  1, 0, (float*)&_modelViewMatrix);
   glUniformMatrix3fv(_nmLoc,  1, 0, (float*)&_modelViewMatrix.inverseTransposed());
   glUniformMatrix4fv(_mvpLoc, 1, 0, (float*)&mvp);

   // Pass lighting uniforms variables
   glUniform4fv(_globalAmbiLoc, 1, (float*)&_globalAmbi);
   glUniform3fv(_lightPosLoc,   1, (float*)&lightPosVS);
   glUniform3fv(_lightDirLoc,   1, (float*)&lightDirVS);
   glUniform4fv(_lightAmbiLoc,  1, (float*)&_lightAmbi);
   glUniform4fv(_lightDiffLoc,  1, (float*)&_lightDiff);
   glUniform4fv(_lightSpecLoc,  1, (float*)&_lightSpec);
   glUniform3fv(_lightAttLoc,   1, (float*)&_lightAtt);
   glUniform4fv(_matAmbiLoc,    1, (float*)&_matAmbi); 
   glUniform4fv(_matDiffLoc,    1, (float*)&_matDiff); 
   glUniform4fv(_matSpecLoc,    1, (float*)&_matSpec); 
   glUniform4fv(_matEmisLoc,    1, (float*)&_matEmis);
   glUniform1f (_matShineLoc, _matShine);
   
   // Pass the active texture unit
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, _textureID0);
   glUniform1i(_texture0Loc, 0);

   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, _textureID1);
   glUniform1i(_texture1Loc, 1);
     
   /////////////////////////////
   // Draw sphere with 2 VBOs //
   /////////////////////////////

   glEnableVertexAttribArray(_pLoc);
   glEnableVertexAttribArray(_nLoc);
   glEnableVertexAttribArray(_tLoc);

   // Acitvate VBOs
   glBindBuffer(GL_ARRAY_BUFFER, _vboV);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _vboI);

   // Activate Texture

   // For VBO only offset instead of data pointer
   GLsizei stride = sizeof(VertexPNT);
   GLsizei offsetN = sizeof(SLVec3f);
   GLsizei offsetT = sizeof(SLVec3f) + sizeof(SLVec3f);
   glVertexAttribPointer(_pLoc, 3, GL_FLOAT, GL_FALSE, stride, 0);
   glVertexAttribPointer(_nLoc, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetN);
   glVertexAttribPointer(_tLoc, 2, GL_FLOAT, GL_FALSE, stride, (void*)offsetT);
   
   // Draw cube with triangles by indexes
   glDrawElements(GL_TRIANGLES, _numI, GL_UNSIGNED_INT, 0);

   // Deactivate buffers
   glBindBuffer(GL_ARRAY_BUFFER, 0);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

   // Disable the vertex arrays
   glDisableVertexAttribArray(_pLoc);
   glDisableVertexAttribArray(_nLoc);
   glDisableVertexAttribArray(_tLoc);
   
   // Fast copy the back buffer to the front buffer. This is OS dependent.
   glfwSwapBuffers();

   // Calculate frames per second
   char title[255];
   static float lastTimeSec = 0;
   float timeNowSec = (float)glfwGetTime();
   float fps = calcFPS(timeNowSec-lastTimeSec);
   sprintf(title, "Texture Mapping, %d x %d, fps: %4.0f", _resolution, _resolution, fps);
   glfwSetWindowTitle(title);
   lastTimeSec = timeNowSec;

   // Check for errors from time to time
   //SL_GET_GL_ERROR;

   // Return true to get an immediate refresh 
   return true;
}
//-----------------------------------------------------------------------------
/*!
onResize: Event handler called on the resize event of the window. This event
should called once before the onPaint event. Do everything that is dependent on
the size and ratio of the window.
*/
void onResize(int width, int height)
{  
   double w = (double)width;
   double h = (double)height;
   
   // define the projection matrix
   _projectionMatrix.perspective(45, w/h, 0.01f, 10.0f);
   
   // define the viewport
   glViewport(0, 0, width, height);

   onPaint();
}
//-----------------------------------------------------------------------------
/*!
Mouse button down & release eventhandler starts and end mouse rotation
*/
void onMouseButton(int button, int state)
{
   SLint x = _mouseX;
   SLint y = _mouseY;
      
   _mouseDown = (state==GLFW_PRESS);
   if (_mouseDown)
   {  _startX = x;
      _startY = y;

      // Renders only the lines of a polygon during mouse moves
      if (button==GLFW_MOUSE_BUTTON_RIGHT)
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
   } else
   {  _rotX += _deltaX;
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
void onMouseMove(int x, int y)
{     
   _mouseX  = x;
   _mouseY  = y;
   
   if (_mouseDown)
   {  _deltaY = x - _startX;
      _deltaX = y - _startY;
      onPaint();
   }
}
//-----------------------------------------------------------------------------
/*!
Mouse wheel eventhandler that moves the camera foreward or backwards
*/
void onMouseWheel(int wheelPos)
{  
   static int lastMouseWheelPos = 0;
   int delta = wheelPos-lastMouseWheelPos;
   lastMouseWheelPos = wheelPos;

   if (_modifiers == NONE)
   {  
      if (delta > 0) _camZ += 0.1f;
      else _camZ -= 0.1f;
      onPaint();
   }
}
//-----------------------------------------------------------------------------
/*!
Key action eventhandler handles key down & release events
*/
void onKey(int GLFWKey, int action)
{         
   if (action==GLFW_PRESS)
   {  
      switch (GLFWKey)
      {
         case GLFW_KEY_ESC:
            onClose();
            glfwCloseWindow();
            break;
         case GLFW_KEY_UP:
            _resolution = _resolution<<1;
            buildSphere(1.0f, _resolution, _resolution);
            break;
         case GLFW_KEY_DOWN:
            if (_resolution > 4) _resolution = _resolution>>1;
            buildSphere(1.0f, _resolution, _resolution);
            break;
         case GLFW_KEY_LSHIFT: _modifiers = _modifiers|SHIFT; break;
         case GLFW_KEY_RSHIFT: _modifiers = _modifiers|SHIFT; break;
         case GLFW_KEY_LCTRL:  _modifiers = _modifiers|CTRL; break; 
         case GLFW_KEY_RCTRL:  _modifiers = _modifiers|CTRL; break; 
         case GLFW_KEY_LALT:   _modifiers = _modifiers|ALT; break;
         case GLFW_KEY_RALT:   _modifiers = _modifiers|ALT; break;
      }
   } else
   if (action == GLFW_RELEASE)
   {  switch (GLFWKey)
      {  case GLFW_KEY_LSHIFT: _modifiers = _modifiers&~SHIFT; break;
         case GLFW_KEY_RSHIFT: _modifiers = _modifiers&~SHIFT; break;
         case GLFW_KEY_LCTRL:  _modifiers = _modifiers&~CTRL; break; 
         case GLFW_KEY_RCTRL:  _modifiers = _modifiers&~CTRL; break; 
         case GLFW_KEY_LALT:   _modifiers = _modifiers&~ALT; break;
         case GLFW_KEY_RALT:   _modifiers = _modifiers&~ALT; break;
      }
   }
}

//-----------------------------------------------------------------------------
/*!
The C main procedure running the GLFW GUI application.
*/
int main()
{  
   // Init the GLFW library for the GUI interface
   if (!glfwInit())
   {  fprintf(stderr, "Failed to initialize GLFW\n");
      exit(1);
   }
   
   // Enable fullscreen anti aliasing with 4 samples
   glfwOpenWindowHint(GLFW_FSAA_SAMPLES, 4);

   // Create the window with the GLFW library
   if (!glfwOpenWindow(512, 512, 0,0,0,0, 32, 0, GLFW_WINDOW))
   {  glfwTerminate();
      fprintf(stderr, "Failed to create GLFW window");
      exit(1);
   }
   
   cout << "\n--------------------------------------------------------------\n";
   cout << "Sphere with ambient, diffuse & specular lighting ...\n";
      
   // Init the GLEW library for the OpenGL functions
   GLenum err = glewInit();
   if (GLEW_OK != err)
   {  fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
      return -1;
   }

   onInit();
 
   glfwEnable(GLFW_MOUSE_CURSOR);            // Show the cursor
   glfwSwapInterval(0);                      // 1=Enable vertical sync

   // Set callback functions
   glfwSetWindowSizeCallback (onResize);
   glfwSetMouseButtonCallback(onMouseButton);
   glfwSetMousePosCallback   (onMouseMove);
   glfwSetMouseWheelCallback (onMouseWheel);
   glfwSetKeyCallback        (onKey);
   glfwSetWindowCloseCallback(onClose);
   
   // The never ending event loop
   while (glfwGetWindowParam(GLFW_OPENED) == GL_TRUE)
   {
      // if no updated occured wait for the next event (power saving)
      if (!onPaint()) 
         glfwWaitEvents(); 
   }

   glfwTerminate();
   exit(0);
}
//-----------------------------------------------------------------------------
