//#############################################################################
//  File:      TextureMapping_Net.cs
//  Purpose:   Minimal core profile OpenGL application for ambient-diffuse-
//             specular lighting shaders with Textures.
//  Author:    Marcus Hudritsch
//  Date:      February 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

using System;
using System.Windows.Forms;
using System.Threading;
using System.Drawing;

using OpenTK;
using gl = OpenTK.Graphics.OpenGL.GL;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;

// Application class inheriting OpenTK's GameWindow
public class TextureMapping_Net : GameWindow
{  
   #region Private Members
   // GLobal application variables
   SLMat4f  _modelViewMatrix;    // 4x4 modelview matrix
   SLMat4f  _projectionMatrix;   // 4x4 projection matrix

   int      _numI = 0;           // NO. of vertex indexes for triangles
   int      _vboV = 0;           // ID of the VBO for vertex array
   int      _vboI = 0;           // ID of the VBO for vertex index array

   float    _camZ;               // z-distance of camera
   float    _rotX, _rotY;        // rotation angles around x & y axis
   int      _deltaX, _deltaY;    //!< delta mouse motion
   int      _startX, _startY;    // x,y mouse start positions
   bool     _mouseLeftDown;      // Flag if mouse is down

   SLVec4f  _globalAmbi;         // global ambient intensity
   SLVec3f  _lightPos;           // Light position in world space
   SLVec3f  _lightDir;           // Light direction in world space
   SLVec4f  _lightAmbient;       // Light ambient intensity   
   SLVec4f  _lightDiffuse;       // Light diffuse intensity   
   SLVec4f  _lightSpecular;      // Light specular intensity
   SLVec4f  _matAmbient;         // Material ambient reflection coeff.
   SLVec4f  _matDiffuse;         // Material diffuse reflection coeff.
   SLVec4f  _matSpecular;        // Material specular reflection coeff.
   SLVec4f  _matEmissive;        // Material emissive coeff.
   float    _matShininess;       // Material shininess exponent

   int      _shaderVertID = 0;   // vertex shader id
   int      _shaderFragID = 0;   // fragment shader id
   int      _shaderProgID = 0;   // shader program id
   int      _textureID = 0;      // texture id
  
   int      _pLoc;               // attribute location for vertex position
   int      _nLoc;               // attribute location for vertex normal
   int      _tLoc;               // attribute location for vertex texcoords
   int      _mvpMatrixLoc;       // uniform location for modelview-projection matrix
   int      _mvMatrixLoc;        // uniform location for modelview matrix
   int      _nMatrixLoc;         // uniform location for normal matrix
   int      _globalAmbiLoc;      // uniform location for global ambient intensity
   int      _lightPosVSLoc;      // uniform location for light position in VS 
   int      _lightDirVSLoc;      // uniform location for light direction in VS 
   int      _lightAmbientLoc;    // uniform location for ambient light intensity 
   int      _lightDiffuseLoc;    // uniform location for diffuse light intensity 
   int      _lightSpecularLoc;   // uniform location for specular light intensity
   int      _matAmbientLoc;      // uniform location for ambient light reflection
   int      _matDiffuseLoc;      // uniform location for diffuse light reflection
   int      _matSpecularLoc;     // uniform location for specular light reflection
   int      _matEmissiveLoc;     // uniform location for light emission
   int      _matShininessLoc;    // uniform location for shininess

   int      _texture0Loc;        // uniform location for texture 0
   
   const float DEG2RAD = (float)(Math.PI/180);
   const float RAD2DEG = (float)(180/Math.PI);
   #endregion
   
   /// <summary>
   /// Contructor that passes parameters to the OpenTK GameWindow
   /// </summary>
   public TextureMapping_Net(int width, 
                             int height, 
                             OpenTK.Graphics.GraphicsMode mode, 
                             string title) : base(width, height, mode, title)
   {
      this.VSync = VSyncMode.On; 
   }

   /// <summary>
   /// BuildSquare creates the vertex attributes for a textured square and VBO.
   /// </summary>
   void BuildSquare()
   {
      // create vertex array for interleaved position, normal and texCoord
	   //                       Position         ,  Normal           ,  texCoord   ,
      float[] v = new float[]{-1.0f, 0.0f, -1.0f,  0.0f, -1.0f, 0.0f,  0.0f,  0.0f, // Vertex 0
                               1.0f, 0.0f, -1.0f,  0.0f, -1.0f, 0.0f,  1.0f,  0.0f, // Vertex 1
                               1.0f, 0.0f,  1.0f,  0.0f, -1.0f, 0.0f,  1.0f,  1.0f, // Vertex 2
                              -1.0f, 0.0f,  1.0f,  0.0f, -1.0f, 0.0f,  0.0f,  1.0f};// Vertex 3

      _vboV = glUtils.BuildVBO(v, 6, 8, 
                               sizeof(float), 
                               BufferTarget.ArrayBuffer, 
                               BufferUsageHint.StaticDraw);

      // create index array for GL_TRIANGLES
      _numI = 6;
      uint[] i = new uint[]{0, 1, 2,  0, 2, 3};
      _vboI = glUtils.BuildVBO(i, _numI, 1, 
                               sizeof(uint), 
                               BufferTarget.ElementArrayBuffer, 
                               BufferUsageHint.StaticDraw);
   }
   
   /// <summary>
   /// OnLoad is called once at the beginning for OpenGL inits.
   /// </summary>
   /// <param name="e"></param>
   protected override void OnLoad(EventArgs e)
   {
      base.OnLoad(e);

      BuildSquare();

      _modelViewMatrix  = new SLMat4f();
      _projectionMatrix = new SLMat4f();

      // Set light parameters
      _globalAmbi    = new SLVec4f(0.0f, 0.0f, 0.0f);
      _lightPos      = new SLVec3f(0.0f, 0.0f, 100.0f);   
      _lightDir      = new SLVec3f(0.0f, 0.0f,-1.0f);   
      _lightAmbient  = new SLVec4f(0.1f, 0.1f, 0.1f);  
      _lightDiffuse  = new SLVec4f(1.0f, 1.0f, 1.0f);
      _lightSpecular = new SLVec4f(1.0f, 1.0f, 1.0f);
      _matAmbient    = new SLVec4f(1.0f, 1.0f, 1.0f);    
      _matDiffuse    = new SLVec4f(1.0f, 1.0f, 1.0f);    
      _matSpecular   = new SLVec4f(1.0f, 1.0f, 1.0f);    
      _matEmissive   = new SLVec4f(0.0f, 0.0f, 0.0f);
      _matShininess  = 100.0f; 
   
      // backwards movement of the camera
      _camZ = -3.0f;      

      // Mouse rotation paramters
      _rotX = 0;
      _rotY = 0;
      _deltaX = 0;
      _deltaY = 0;
      _mouseLeftDown = false;

      // Load textures
      _textureID = glUtils.BuildTexture("../_data/images/textures/earth2048_C.jpg",
                                        TextureMinFilter.LinearMipmapLinear,
                                        TextureMagFilter.Linear,
                                        TextureWrapMode.Repeat,
                                        TextureWrapMode.Repeat);

      // Load, compile & link shaders
      _shaderVertID = glUtils.BuildShader("../lib-SLProject/source/oglsl/ADSTex.vert", ShaderType.VertexShader);
      _shaderFragID = glUtils.BuildShader("../lib-SLProject/source/oglsl/ADSTex.frag", ShaderType.FragmentShader);
      _shaderProgID = glUtils.BuildProgram(_shaderVertID, _shaderFragID);

      // Activate the shader programm
      gl.UseProgram(_shaderProgID); 

      // Get the variable locations (identifiers) within the program
      _pLoc            = gl.GetAttribLocation (_shaderProgID, "a_position");
      _nLoc            = gl.GetAttribLocation (_shaderProgID, "a_normal");
      _tLoc            = gl.GetAttribLocation (_shaderProgID, "a_texCoord");
      _mvMatrixLoc     = gl.GetUniformLocation(_shaderProgID, "u_mvMatrix");
      _mvpMatrixLoc    = gl.GetUniformLocation(_shaderProgID, "u_mvpMatrix");
      _nMatrixLoc      = gl.GetUniformLocation(_shaderProgID, "u_nMatrix");
      _globalAmbiLoc   = gl.GetUniformLocation(_shaderProgID, "u_globalAmbi");
      _lightPosVSLoc   = gl.GetUniformLocation(_shaderProgID, "u_lightPosVS");
      _lightDirVSLoc   = gl.GetUniformLocation(_shaderProgID, "u_lightDirVS");
      _lightAmbientLoc = gl.GetUniformLocation(_shaderProgID, "u_lightAmbient");
      _lightDiffuseLoc = gl.GetUniformLocation(_shaderProgID, "u_lightDiffuse");
      _lightSpecularLoc= gl.GetUniformLocation(_shaderProgID, "u_lightSpecular");
      _matAmbientLoc   = gl.GetUniformLocation(_shaderProgID, "u_matAmbient");
      _matDiffuseLoc   = gl.GetUniformLocation(_shaderProgID, "u_matDiffuse");
      _matSpecularLoc  = gl.GetUniformLocation(_shaderProgID, "u_matSpecular");
      _matEmissiveLoc  = gl.GetUniformLocation(_shaderProgID, "u_matEmissive");
      _matShininessLoc = gl.GetUniformLocation(_shaderProgID, "u_matShininess");
      _texture0Loc     = gl.GetUniformLocation(_shaderProgID, "u_texture0");      

      // Set some OpenGL states
      gl.ClearColor(0.0f, 0.0f, 0.0f, 1);  // Set the background color         
      gl.Enable(EnableCap.DepthTest);      // Enables depth test
      gl.Enable(EnableCap.CullFace);       // Enables the culling of back faces
      
      // Attach mouse wheel handler
      Mouse.WheelChanged += new EventHandler<OpenTK.Input.MouseWheelEventArgs>(Mouse_WheelChanged);
      
      glUtils.GetGLError("OnLoad", true);
   }
   
   /// <summary>
   /// OnResize is called whenever the window resizes
   /// </summary>
   /// <param name="e"></param>
   protected override void OnResize(EventArgs e)
   {
      base.OnResize(e);

      float w = (float)this.Width;
      float h = (float)this.Height;
   
      // define the projection matrix
      _projectionMatrix.Perspective(45, w/h, 0.01f, 10.0f);
   
      // define the viewport
      gl.Viewport(0, 0, (int)w, (int)h);

      glUtils.GetGLError("OnResize", true);
   }

   /// <summary>
   /// Event handler for mouse wheel event
   /// </summary>
   /// <param name="sender">event sender object</param>
   /// <param name="e">event arguments</param>
   void Mouse_WheelChanged(object sender, OpenTK.Input.MouseWheelEventArgs e)
   {
      // Update z-Distance of camera
      _camZ -= (float)e.Delta;
   }   
   
   /// <summary>
   /// OnUpdateFrame is called once before OnRenderFrame
   /// </summary>
   /// <param name="e">event arguments</param>
   protected override void OnUpdateFrame(FrameEventArgs e)
   {
      base.OnUpdateFrame(e);

      if (Keyboard[Key.Escape]) 
         this.Exit( );
      
      if (Mouse[MouseButton.Left])
      {  
         if (!_mouseLeftDown)
         {  _mouseLeftDown = true;        // flag for mouse down
            _startX = Mouse.X;            // keep start position
            _startY = Mouse.Y;
         }
         _deltaY = Mouse.X-_startX;       // calculate delta angle
         _deltaX = Mouse.Y-_startY;
          
      } else
      {  
         if (_mouseLeftDown)
         {  _mouseLeftDown = false;
            _rotX += _deltaX;             // increment rotation with
            _rotY += _deltaY;             // delta angles
            _deltaX = 0;
            _deltaY = 0;
         }
      }
   }

   /// <summary>
   /// OnRenderFrame is called on every frame for rendering 
   /// </summary>
   /// <param name="e">event arguments</param>
   protected override void OnRenderFrame(FrameEventArgs e)
   {
      base.OnRenderFrame(e);
      
      // Clear the color & depth buffer
      gl.Clear(ClearBufferMask.ColorBufferBit | 
               ClearBufferMask.DepthBufferBit);

      // Start with identity every frame
      _modelViewMatrix.Identity();
   
      // View transform: move the coordinate system away from the camera
      _modelViewMatrix.Translate(0, 0, _camZ);

      // View transform: rotate the coordinate system increasingly
      _modelViewMatrix.Rotate(_rotX + _deltaX, 1,0,0);
      _modelViewMatrix.Rotate(_rotY + _deltaY, 0,1,0);

      // Transform light position & direction into view space
      SLVec3f lightPosVS = _modelViewMatrix * _lightPos;
   
      // The light dir is not a position. We only take the rotation of the mv matrix.
      SLMat3f viewRot    = _modelViewMatrix.Mat3();
      SLVec3f lightDirVS = viewRot * _lightDir;

      // Rotate the model so that we see it
      _modelViewMatrix.Rotate(90, -1,0,0);

      // Build the combined modelview-projection matrix
      SLMat4f mvp = new SLMat4f(_projectionMatrix);
      mvp.Multiply(_modelViewMatrix);

      // Build normal matrix
      SLMat3f nm = _modelViewMatrix.InverseTransposed();

      // Pass the matrix uniform variables
      unsafe
      {  gl.UniformMatrix4(_mvMatrixLoc,  1, false, _modelViewMatrix.m);
         gl.UniformMatrix3(_nMatrixLoc,   1, false, nm.m);
         gl.UniformMatrix4(_mvpMatrixLoc, 1, false, mvp.m);

         // Pass lighting uniforms variables
         gl.Uniform4(_globalAmbiLoc,     1, (float[])_globalAmbi);
         gl.Uniform3(_lightPosVSLoc,     1, (float[])lightPosVS);
         gl.Uniform3(_lightDirVSLoc,     1, (float[])lightDirVS);
         gl.Uniform4(_lightAmbientLoc,   1, (float[])_lightAmbient);
         gl.Uniform4(_lightDiffuseLoc,   1, (float[])_lightDiffuse);
         gl.Uniform4(_lightSpecularLoc,  1, (float[])_lightSpecular);
         gl.Uniform4(_matAmbientLoc,     1, (float[])_matAmbient); 
         gl.Uniform4(_matDiffuseLoc,     1, (float[])_matDiffuse); 
         gl.Uniform4(_matSpecularLoc,    1, (float[])_matSpecular); 
         gl.Uniform4(_matEmissiveLoc,    1, (float[])_matEmissive);
      }
      gl.Uniform1(_matShininessLoc,   _matShininess);
      gl.Uniform1(_texture0Loc,       0);
     
      //////////////////////
      // Draw with 2 VBOs //
      //////////////////////

      // Enable all of the vertex attribute arrays
      gl.EnableVertexAttribArray(_pLoc);
      gl.EnableVertexAttribArray(_nLoc);
      gl.EnableVertexAttribArray(_tLoc);

      // Activate VBOs
      gl.BindBuffer(BufferTarget.ArrayBuffer, _vboV);
      gl.BindBuffer(BufferTarget.ElementArrayBuffer, _vboI);

      // Activate Texture
      gl.BindTexture(TextureTarget.Texture2D, _textureID);

      // For VBO only offset instead of data pointer
      int stride  = 32;
      int offsetN = 3 * sizeof(float);
      int offsetT = 6 * sizeof(float);
      gl.VertexAttribPointer(_pLoc, 3, VertexAttribPointerType.Float, false, stride, 0);
      gl.VertexAttribPointer(_nLoc, 3, VertexAttribPointerType.Float, false, stride, offsetN);
      gl.VertexAttribPointer(_tLoc, 2, VertexAttribPointerType.Float, false, stride, offsetT);
   
      /////////////////////////////////////////////////////////////////////////////
      // Draw cube model triangles by indexes
      gl.DrawElements(BeginMode.Triangles, _numI, DrawElementsType.UnsignedInt, 0);
      /////////////////////////////////////////////////////////////////////////////

      // Deactivate buffers
      gl.BindBuffer(BufferTarget.ArrayBuffer, 0);
      gl.BindBuffer(BufferTarget.ElementArrayBuffer, 0);

      // Disable the vertex arrays
      gl.DisableVertexAttribArray(_pLoc);
      gl.DisableVertexAttribArray(_nLoc);
      gl.DisableVertexAttribArray(_tLoc);
      
      // Fast copy the back buffer to the front buffer. This is OS dependent.
      SwapBuffers();
      
      // Check for errors
      glUtils.GetGLError("OnRenderFrame", true);
   }

   /// <summary>
   /// Entry point of this application
   /// </summary>
   public static void Main()
   {  
      // Print some help info
      Console.WriteLine("\n--------------------------------------------------------------");
      Console.WriteLine("Texture Mapping with OpenGL and DotNet");
   
      OpenTK.Graphics.GraphicsMode mode = 
         new OpenTK.Graphics.GraphicsMode(32, // bits color buffer
                                          24, // bits depth buffer
                                          0,  // bits stencil buffer
                                          4); // No. of sample for AA
                             
      TextureMapping_Net app = new TextureMapping_Net(640, 480, mode, "OpenGL Texture Mapping (.NET)");
      app.Run(30.0, 0.0);
   }
}
