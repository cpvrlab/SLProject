//#############################################################################
//  File:      Globals/GL/glUtils.cs
//  Purpose:   General OpenGL utility functions for simple OpenGL demo apps
//  Date:      February 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

using System;
using System.IO;
using System.Windows.Forms;
using System.Drawing;
using System.Drawing.Imaging;

using OpenTK.Graphics.OpenGL;
using gl = OpenTK.Graphics.OpenGL.GL;

/// <summary>
/// OpenGL Utility functions
/// </summary>
public class glUtils
{
   /// <summary>
   /// Loads the ASCII content of a shader file and returns it as a string.
   /// If the file can not be opened an error message is sent to stdout before the app
   /// exits with code 1.
   /// </summary>
   /// <param name="filename">Full path and file name</param>
   public static string LoadShader(string filename)
   {
      try
      {  return File.ReadAllText(filename);
      }
      catch (Exception ex) 
      {  Console.WriteLine("LoadShader failed: " + ex.ToString());
         Application.Exit();
      }
      return "";
   }

   /// <summary>
   /// Load the shader file, creates an OpenGL shader object, compiles the 
   /// source code and returns the handle to the internal shader object. If the 
   /// compilation fails the compiler log is sent to the console before the app exits 
   /// with code 1.
   /// </summary>
   /// <param name="shaderFile">Full path and file name of the shader file</param>
   /// <param name="shaderType">OpenTK.Graphics.OpenGL.ShaderType</param>
   /// <returns></returns>
   public static int BuildShader(string shaderFile, ShaderType shaderType)
   {  
      // Load shader file, create shader and compile it
      string source = LoadShader(shaderFile);

      int shaderHandle = gl.CreateShader(shaderType);
      gl.ShaderSource(shaderHandle, source);
      gl.CompileShader(shaderHandle);
   
      // Check compile success
      int compileSuccess;
		gl.GetShader(shaderHandle, ShaderParameter.CompileStatus, out compileSuccess);
		if(compileSuccess == 0) 
      {  String message;
			gl.GetShaderInfoLog(shaderHandle, out message);
         Console.WriteLine("BuildShader failed: " + message);
         Application.Exit();
		}
      GetGLError("BuildShader", true);
      return shaderHandle;
   }

   /// <summary>
   /// Creates a program object, attaches the shaders, links them and 
   /// returns the OpenGL handle of the program. If the linking fails the linker log 
   /// is sent to the stdout before the app exits with code 1.
   /// </summary>
   /// <param name="vertShaderID"></param>
   /// <param name="fragShaderID"></param>
   /// <returns></returns>
   public static int BuildProgram(int vertShaderID, int fragShaderID)
   {  
      // Create program, attach shaders and link them 
      int programHandle = gl.CreateProgram();
      gl.AttachShader(programHandle, vertShaderID);
      gl.AttachShader(programHandle, fragShaderID);
      gl.LinkProgram(programHandle);
   
      // Check linker success
      int linkSuccess;
      gl.GetProgram(programHandle, ProgramParameter.LinkStatus, out linkSuccess);
      if(linkSuccess == 0) 
      {  String message;
		   gl.GetProgramInfoLog(programHandle, out message);
			Console.WriteLine("Program link failed: " + message);
	   }
		
      // Validate program
      int validateSuccess;
		gl.ValidateProgram(programHandle);
		gl.GetProgram(programHandle, ProgramParameter.ValidateStatus, out validateSuccess);
		if(validateSuccess == 0) 
      {  String message;
			gl.GetProgramInfoLog(programHandle, out message);
			Console.WriteLine("Program validation failed", message);
		}
      
      GetGLError("BuildProgram", true);
      return programHandle;
   }

   /// <summary>
   /// Generates a Vertex Buffer Object (VBO) and copies the data into the
   /// buffer on the GPU and returns the id of the buffer,
   /// The size of the buffer is calculated as numElements * 
   /// elementSize * typeSize which means e.g.(NO. of vertices) * (3 for x,y& z) * 
   /// (4 for float). The targetTypeGL distincts between GL_ARRAY_BUFFER for attribute 
   /// data and GL_ELEMENT_ARRAY_BUFFER for index data. The usageTypeGL distincts 
   /// between GL_STREAM_DRAW, GL_STATIC_DRAW and GL_DYNAMIC_DRAW.
   /// </summary>
   public static int BuildVBO(object data, 
                              int numElements, 
                              int elementSize, 
                              int typeSize,        
                              BufferTarget targetTypeGL,
                              BufferUsageHint usageTypeGL)
   {  
      // Generate a buffer id
      int vboID;
      gl.GenBuffers(1, out vboID);
   
      // Binds (activates) the buffer that is used next
      gl.BindBuffer(targetTypeGL, vboID);
   
      // determine the buffersize in bytes
      int bufSize = numElements * elementSize * typeSize;
   
      // Copy data to the VBO on the GPU. The data could be delete afterwards.
      if (data is float[])
         gl.BufferData<float>(targetTypeGL, new IntPtr(bufSize), (float[])data, usageTypeGL);
      else if (data is int[])
         gl.BufferData<int>(targetTypeGL, new IntPtr(bufSize), (int[])data, usageTypeGL);
      else if (data is uint[])
         gl.BufferData<uint>(targetTypeGL, new IntPtr(bufSize), (uint[])data, usageTypeGL);

      GetGLError("BuildVBO", true);
      return vboID;
   }

   /// <summary>
   /// buildTexture loads and build the OpenGL texture on the GPU. The loaded image
   /// data in the client memory is deleted again. The parameters min_filter and
   /// mag_filter set the minification and magnification. The wrapS and wrapT parameters
   /// set the texture wrapping mode. See the GL spec.
   /// </summary>
   /// <param name="textureFile"></param>
   /// <param name="min_filter"></param>
   /// <param name="mag_filter"></param>
   /// <param name="wrapS"></param>
   /// <param name="wrapT"></param>
   /// <returns></returns>
   public static int BuildTexture(string textureFile,
                                  TextureMinFilter min_filter,
                                  TextureMagFilter mag_filter,
                                  TextureWrapMode wrapS, 
                                  TextureWrapMode wrapT)
   {  
      // load texture image
      Bitmap img = null;
      try {img = (Bitmap)Image.FromFile(textureFile);}
      catch(Exception ex)
      {  Console.WriteLine("BuildTexture: Loading failed: " + ex.ToString());
         Application.Exit();
      }

      // Images on Windows are top left but OpenGL expects them bottom left
      img.RotateFlip(RotateFlipType.RotateNoneFlipY);

      // check max. size
      int maxSize = 0;
      gl.GetInteger(GetPName.MaxTextureSize, out maxSize);
      if (img.Width  > maxSize || img.Height > maxSize) 
      {  Console.WriteLine("BuildTexture: Texture width or height is too big.");
         Application.Exit();
      }

      // generate texture name (= internal texture object)
      int textureHandle;
      gl.GenTextures(1, out textureHandle);

      // bind the texture as the active one
		gl.BindTexture(TextureTarget.Texture2D, textureHandle);

      // apply minification & magnification filter
      gl.TexParameter(TextureTarget.Texture2D, 
                      TextureParameterName.TextureMinFilter, 
                      (int)min_filter);
      gl.TexParameter(TextureTarget.Texture2D, 
                      TextureParameterName.TextureMagFilter, 
                      (int)mag_filter);
      
      // apply texture wrapping modes
      gl.TexParameter(TextureTarget.Texture2D, 
                      TextureParameterName.TextureWrapS, 
                      (int)wrapS);
      gl.TexParameter(TextureTarget.Texture2D, 
                      TextureParameterName.TextureWrapT, 
                      (int)wrapT);
      
      // Lock the image data memory to access the data
      BitmapData data = img.LockBits(new Rectangle(0, 0, img.Width, img.Height), 
                                     ImageLockMode.ReadOnly, 
                                     img.PixelFormat);

      PixelInternalFormat internFormat = PixelInternalFormat.Four;
      OpenTK.Graphics.OpenGL.PixelFormat pixelFormat = OpenTK.Graphics.OpenGL.PixelFormat.Bgra;
      switch(img.PixelFormat)
      {  case System.Drawing.Imaging.PixelFormat.Alpha:           
            internFormat = PixelInternalFormat.Alpha; 
            pixelFormat = OpenTK.Graphics.OpenGL.PixelFormat.Alpha;
            break;
         case System.Drawing.Imaging.PixelFormat.Format24bppRgb:  
            internFormat = PixelInternalFormat.Rgb; 
            pixelFormat = OpenTK.Graphics.OpenGL.PixelFormat.Bgr;
            break;
         case System.Drawing.Imaging.PixelFormat.Format8bppIndexed: 
            internFormat = PixelInternalFormat.Luminance;  
            pixelFormat = OpenTK.Graphics.OpenGL.PixelFormat.Luminance;
            break;
      }

                                     
      // Copy image data to the GPU. The image can be delete afterwards
      gl.TexImage2D(TextureTarget.Texture2D, // target texture type 1D, 2D or 3D
                    0,                       // Base level for mipmapped textures
                    internFormat,            // internal format: e.g. GL_RGBA, see spec.
                    img.Width,               // image width
                    img.Height,              // image height
                    0,                       // border pixels: must be 0
                    pixelFormat,             // pixel format
                    PixelType.UnsignedByte,  // data type
                    data.Scan0);             // image data pointer
   
      // generate the mipmap levels 
      if (min_filter>=TextureMinFilter.NearestMipmapNearest)
	   {  gl.GenerateMipmap(GenerateMipmapTarget.Texture2D);
      }

      // Unlock the memory and dispose it
      img.UnlockBits(data);
      img.Dispose();

      GetGLError("BuildTexture", true);
      return textureHandle;
   }

   /// <summary>
   /// Gets the last OpenGL error that occured
   /// </summary>
   /// <param name="location">location string to identify calling place</param>
   /// <param name="quit">exits the app if true</param>
   public static void GetGLError(string location, bool quit)
   {  
      #if DEBUG
      ErrorCode err = gl.GetError();
	   if(err != ErrorCode.NoError) 
      {
         string errStr;
         switch(err)
         {  case ErrorCode.InvalidEnum: 
               errStr = "GL_INVALID_ENUM"; break;
            case ErrorCode.InvalidValue: 
               errStr = "GL_INVALID_VALUE"; break;
            case ErrorCode.InvalidOperation: 
               errStr = "GL_INVALID_OPERATION"; break;
            case ErrorCode.InvalidFramebufferOperation: 
               errStr = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
            case ErrorCode.OutOfMemory: 
               errStr = "GL_OUT_OF_MEMORY"; break;
            default: errStr = "Unknown error"; break;
         }
         Console.WriteLine(location + ": " + errStr);
      
         if (quit) Application.Exit();
      }
      #endif
   }
}