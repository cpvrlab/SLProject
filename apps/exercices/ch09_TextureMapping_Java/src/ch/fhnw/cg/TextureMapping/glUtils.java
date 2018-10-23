//#############################################################################
//  File:      glUtils.java
//  Purpose:   General OpenGL utility functions for simple OpenGL demo apps
//  Author:    Marcus Hudritsch
//  Date:      September 2012 (HS12)
//  Copyright: Marcus Hudritsch, Switzerland
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

package ch.fhnw.cg.TextureMapping;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;

import javax.imageio.ImageIO;
import javax.media.opengl.GL2GL3;

import com.sun.opengl.util.BufferUtil;

/**
 * @author hudrima1
 *
 */
public class glUtils
{		
    /**
     * Loads the ASCII content of a shader file and returns it as a string.
     * If the file can not be opened an error message is sent to System.out 
     * before the application exits with code 1.
     *  
     * @param filename (Full path and name of a file that contains the shader code) 
     * @return 	shader code as single string
     */
    public static String loadShader(String filename)
    {
        String vertCode = "";
        try
        {	String line;
            BufferedReader reader = new BufferedReader(new FileReader(filename));
            while((line=reader.readLine())!=null)
                vertCode += line + "\n";
            reader.close();
            return vertCode;
        }
        catch(Exception e)
        {	System.out.println("File open failed: " + filename);
            System.exit(1);
        }
        return vertCode;
    }
	
    /**
     * Loads the shader file, creates an OpenGL shader object, compiles the 
     * source code and returns the handle to the internal shader object. If the 
     * compilation fails the compiler log is sent to the System.out before the 
     * application exits with code 1.
     * @param gl (OpenGL interface)
     * @param shaderFile (Full path and filename of the shader text file)
     * @param shaderType (GL_VERTEX_SHADER, GL_FRAGMENT_SHADER)
     * @return OpenGL shader id
     */
    public static int buildShader(GL2GL3 gl, String shaderFile, int shaderType)
    {  
        String[] source = new String[1];
        source[0] = loadShader(shaderFile);
      
        int shaderHandle = gl.glCreateShader(shaderType);
        gl.glShaderSource(shaderHandle, 1, source, null);
        gl.glCompileShader(shaderHandle);
        
        // Check compilation success
        int compileSuccess[] = new int[1];
        gl.glGetShaderiv(shaderHandle, GL2GL3.GL_COMPILE_STATUS, IntBuffer.wrap(compileSuccess));
        if(compileSuccess[0] == GL2GL3.GL_FALSE)
        {
            int logLen[] = new int[1];
            gl.glGetShaderiv(shaderHandle, GL2GL3.GL_INFO_LOG_LENGTH, IntBuffer.wrap(logLen));
            ByteBuffer infoLog = BufferUtil.newByteBuffer(logLen[0]); 
            gl.glGetShaderInfoLog(shaderHandle, logLen[0], null, infoLog);
            byte[] infoBytes = new byte[logLen[0]];
            infoLog.get(infoBytes);
            String log = new String(infoBytes);
            System.out.println("**** Compile Error ****");
            System.out.println("In File: " + shaderFile);
            System.out.println(log);
            System.exit(1);
        }
        return shaderHandle;
    }
		
    /**
     * Builds the vertex & fragment shaders, creates a program object,
     * attaches the shaders, links them and returns the OpenGL handle of the 
     * program. If the linking fails the linker log is sent to the System.out 
     * before the application exits with code 1.
     * @param gl (OpenGL interface)
     * @param vertShaderID (Vertex shader id)
     * @param fragShaderID (Fragment shader id)
     * @return OpenGL program id
     */
    public static int buildProgram(GL2GL3 gl,
                                   int vertShaderID, 
                                   int fragShaderID) 
    {
        int programHandle = gl.glCreateProgram();
        gl.glAttachShader(programHandle, vertShaderID);
        gl.glAttachShader(programHandle, fragShaderID);
        gl.glLinkProgram(programHandle);

        // Check link success
        int linkSuccess[] = new int[1];
        gl.glGetProgramiv(programHandle, GL2GL3.GL_LINK_STATUS, IntBuffer.wrap(linkSuccess));
        if (linkSuccess[0] == GL2GL3.GL_FALSE) 
        {
            int logLen[] = new int[1];
            gl.glGetShaderiv(programHandle, GL2GL3.GL_INFO_LOG_LENGTH, IntBuffer.wrap(logLen));
            ByteBuffer infoLog = BufferUtil.newByteBuffer(logLen[0]);
            gl.glGetProgramInfoLog(programHandle, logLen[0], null, infoLog);
            byte[] infoBytes = new byte[logLen[0]];
            infoLog.get(infoBytes);
            String log = new String(infoBytes);
            System.out.println("**** Link Error ****");
            System.out.println(log);
            System.exit(1);
        }
        return programHandle;
    }

    /** 
     * buildVBO generates a Vertex Buffer Object (VBO) and copies the data into the
     * buffer on the GPU and returns the id of the buffer,
     * The size of the buffer is calculated as numElements * 
     * elementSize * typeSize which means e.g.(NO. of vertices) * (3 for x,y& z) * 
     * (4 for float). The targetTypeGL distinct between GL_ARRAY_BUFFER for attribute 
     * data and GL_ELEMENT_ARRAY_BUFFER for index data. The usageTypeGL distinct 
     * between GL_STREAM_DRAW, GL_STATIC_DRAW and GL_DYNAMIC_DRAW.
     * @param gl (OpenGL interface)
     * @param data (Native IO buffer data buffer (IntBuffer or FloatBuffer))
     * @param numElements (NO. of e.g. vertices)
     * @param elementSize (Size of one element in e.g. floats)
     * @param typeSize (Size of type, e.g. bytes per float)
     * @param targetTypeGL (GL_ARRAY_BUFFER or GL_ELEMENT_ARRAY_BUFFER)
     * @param usageTypeGL (GL_STREAM_DRAW, GL_STATIC_DRAW, or GL_DYNAMIC_DRAW)
     * @return VBO id as int[]
     */    
	public static int[] buildVBO(GL2GL3 gl,
							   	 Buffer data, 
							   	 int numElements, 
							   	 int elementSize, 
							   	 int typeSize,        
							   	 int targetTypeGL,
							   	 int usageTypeGL)
	{  
		// Generate a buffer id
		int[] vboID = new int[1];
		gl.glGenBuffers(1, vboID, 0);
		   
		// Binds (activates) the buffer that is used next
		gl.glBindBuffer(targetTypeGL, vboID[0]);
		   
		// determine the buffer size in bytes
		int bufSize = numElements * elementSize * typeSize;
		   
		// Copy data to the VBO on the GPU. The data could be delete afterwards.
		gl.glBufferData(targetTypeGL, bufSize, data, usageTypeGL);
		
		return vboID;
	}
	
	/**
	 * buildTexture loads and build the OpenGL texture on the GPU. The loaded image
	 * data in the client memory is deleted again. The parameters min_filter and
	 * mag_filter set the minification and magnification. The wrapS and wrapT parameters
	 * set the texture wrapping mode. See the GL specification.
	 * @param gl OpenGL interface
	 * @param textureFile Full path and file name of texture image file
	 * @param min_filter GL_NEAREST, GL_LINEAR, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST_MIPMAP_LINEAR, GL_LINEAR_MIPMAP_NEAREST or GL_LINEAR_MIPMAP_LINEAR
	 * @param mag_filter GL_NEAREST or GL_LINEAR
	 * @param wrapS GL_CLAMP_TO_EDGE, GL_MIRRORED_REPEAT, or GL_REPEAT
	 * @param wrapT GL_CLAMP_TO_EDGE, GL_MIRRORED_REPEAT, or GL_REPEAT
	 */
	public static int[] buildTexture(GL2GL3 gl,
			                       	 String textureFile,
			                       	 int min_filter,
			                       	 int mag_filter,
			                       	 int wrapS, 
			                       	 int wrapT)
	{  	int width = 0;
		int height = 0;
		
		try 
		{
			BufferedImage img=ImageIO.read(new File(textureFile));
			width =  img.getWidth();
			height = img.getHeight();
			ByteBuffer buffer=ByteBuffer.allocateDirect(height*width*4);
			IntBuffer  buf=buffer.asIntBuffer();
			
			// Flip the texture image because windows returns the image top left
			// and OpenGL expects it bottom left. 
			// Also make sure the image component are RGBA
			for(int y=height-1; y >=0; y--)
			{	for(int x=0; x<width; x++)
				{	int color = img.getRGB(x,y);
					int blue  =  color & 0x000000ff;
					int green = (color & 0x0000ff00) >>  8;
					int red   = (color & 0x00ff0000) >> 16;
					int alpha = (color & 0xff0000)   >> 24;					
					color = red*0x1000000 + green*0x10000 + blue*0x100 + alpha;
					buf.put(color);
				}	
			}
	
			// check max. size
			int[] maxSize = new int[1];
			gl.glGetIntegerv(GL2GL3.GL_MAX_TEXTURE_SIZE, maxSize, 0);
			if (width  > maxSize[0] || height > maxSize[0]) 
			{  	System.out.println("SLGLTexture::build: Texture height is too big.");
	        	System.exit(1);
			}
	
			// generate texture name (= internal texture object)
			int[] textureHandle = new int[1];
			gl.glGenTextures(1, textureHandle, 0);
		
			// bind the texture as the active one
			gl.glBindTexture(GL2GL3.GL_TEXTURE_2D, textureHandle[0]);
		
			// apply minification & magnification filter
			gl.glTexParameteri(GL2GL3.GL_TEXTURE_2D, GL2GL3.GL_TEXTURE_MIN_FILTER, min_filter);
			gl.glTexParameteri(GL2GL3.GL_TEXTURE_2D, GL2GL3.GL_TEXTURE_MAG_FILTER, mag_filter);
		      
			// apply texture wrapping modes
			gl.glTexParameteri(GL2GL3.GL_TEXTURE_2D, GL2GL3.GL_TEXTURE_WRAP_S, wrapS);
			gl.glTexParameteri(GL2GL3.GL_TEXTURE_2D, GL2GL3.GL_TEXTURE_WRAP_T, wrapT);
	
			// Copy image data to the GPU. The image can be delete afterwards
			gl.glTexImage2D(GL2GL3.GL_TEXTURE_2D,  	// target texture type 1D, 2D or 3D
		            		0,               		// Base level for mipmapped textures
		            		GL2GL3.GL_RGBA,    		// internal format: e.g. GL_RGBA, see spec.
		            		width,     				// image width
		            		height,    				// image height
		            		0,               		// border pixels: must be 0
		            		GL2GL3.GL_RGBA,    		// data format: e.g. GL_RGBA, see spec. 
		            		GL2GL3.GL_UNSIGNED_BYTE,// data type
		            		buffer); 				// image data pointer
	   
			// generate the mipmap levels 
			if (min_filter >= GL2GL3.GL_NEAREST_MIPMAP_NEAREST)
				gl.glGenerateMipmap(GL2GL3.GL_TEXTURE_2D);

			   
			return textureHandle;

		
		} catch (Exception e) 
		{
			e.printStackTrace();
            System.exit(1);
            return null;
		} 
	}

	/**
	 * Print the OpenGL error if any occurred.
	 * @param gl: OpenGL interface
	 * @param location: Additional information string 
	 */
	public static void getGLError(GL2GL3 gl, String location)
	{
		int errCode = gl.glGetError();
		if (errCode == GL2GL3.GL_NO_ERROR) return;
		
		String errStr;
		switch(errCode)
		{  	case GL2GL3.GL_INVALID_ENUM: 
	        	errStr = "GL_INVALID_ENUM"; break;
			case GL2GL3.GL_INVALID_VALUE: 
				errStr = "GL_INVALID_VALUE"; break;
			case GL2GL3.GL_INVALID_OPERATION: 
				errStr = "GL_INVALID_OPERATION"; break;
			case GL2GL3.GL_INVALID_FRAMEBUFFER_OPERATION: 
				errStr = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
			case GL2GL3.GL_OUT_OF_MEMORY: 
				errStr = "GL_OUT_OF_MEMORY"; break;
			default: errStr = "Unknown error";
		}
		System.out.println(location + ": " + errStr);
	}
}
