package ch.fhnw.cg.TextureMapping;

import java.awt.BorderLayout;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import javax.media.opengl.GL2GL3;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.GLProfile;
import javax.media.opengl.awt.GLCanvas;
import javax.swing.JFrame;


public class TextureMapping 
{
	static JFrame frame;
	
	/**
	 * The applications main routine
	 * @param args: command line arguments
	 */
	public static void main(String[] args) 
	{				
		// Request multisampling capability for antialiasing
		GLCapabilities caps = new GLCapabilities(GLProfile.getDefault());
		caps.setSampleBuffers(true); 	// Activate multisampling
		caps.setNumSamples(4);			// set multisampling samples
		
		// Create GUI window with a Swing JFrame
		frame = new JFrame("OpenGL Texture Mapping Example");
		frame.getContentPane().setLayout(new BorderLayout());
		frame.getContentPane().add(new JOGLCanvas(caps), BorderLayout.CENTER);
		frame.pack();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		frame.setSize(640, 480);
	} 
	
	/**
	 * The OpenGL interface class based on GLCanvas
	 */
	@SuppressWarnings("serial")
	private static class JOGLCanvas extends GLCanvas implements GLEventListener 
	{  
		private SLMat4f _modelViewMatrix;  	// 4x4 modelview matrix
		private SLMat4f _projectionMatrix; 	// 4x4 projection matrix
		
		private int    	_numI;              // NO. of vertex indexes for triangles
		private int[] 	_vboV;              // ID of the VBO for vertex array
		private int[]   _vboI;              // ID of the VBO for vertex index array
		
		private float   _camZ;              // z-distance of camera
		private float   _rotX, _rotY;       // rotation angles around x & y axis
		private int     _deltaX, _deltaY;   // delta mouse motion
		private int     _startX, _startY;   // x,y mouse start positions
		private boolean	_mouseDown;     	// Flag if mouse is down
		private int     _polygonMode;       // OpenGL polygon mode
		
		private SLVec4f	_globalAmbi;        // global ambient intensity
		private SLVec3f _lightPos;          // Light position in world space
		private SLVec3f	_lightDir;          // Light direction in world space
		private SLVec4f	_lightAmbient;      // Light ambient intensity   
		private SLVec4f	_lightDiffuse;      // Light diffuse intensity   
		private SLVec4f	_lightSpecular;     // Light specular intensity
		private SLVec4f	_matAmbient;        // Material ambient reflection coeff.
		private SLVec4f	_matDiffuse;        // Material diffuse reflection coeff.
		private SLVec4f	_matSpecular;       // Material specular reflection coeff.
		private SLVec4f	_matEmissive;       // Material emissive coeff.
		private float	_matShininess;      // Material shininess exponent

		private int   	_shaderVertID;      // vertex shader id
		private int   	_shaderFragID;      // fragment shader id
		private int   	_shaderProgID;      // shader program id
		private int[] 	_textureID;         // texture id
   
		private int    	_pLoc;            	// attribute location for vertex position
		private int    	_nLoc;            	// attribute location for vertex normal
		private int    	_tLoc;            	// attribute location for vertex texcoords
		private int    	_mvpMatrixLoc;    	// uniform location for modelview-projection matrix
		private int    	_mvMatrixLoc;    	// uniform location for modelview matrix
		private int    	_nMatrixLoc;      	// uniform location for normal matrix

		private int    	_globalAmbiLoc;   	// uniform location for global ambient intensity
		private int    	_lightPosVSLoc;   	// uniform location for light position in VS 
		private int    	_lightDirVSLoc;   	// uniform location for light direction in VS 
		private int    	_lightAmbientLoc; 	// uniform location for ambient light intensity 
		private int    	_lightDiffuseLoc; 	// uniform location for diffuse light intensity 
		private int    	_lightSpecularLoc;	// uniform location for specular light intensity
		private int    	_matAmbientLoc;   	// uniform location for ambient light reflection
		private int    	_matDiffuseLoc;   	// uniform location for diffuse light reflection
		private int    	_matSpecularLoc;  	// uniform location for specular light reflection
		private int    	_matEmissiveLoc;  	// uniform location for light emission
		private int    	_matShininessLoc; 	// uniform location for shininess

		private int    	_texture0Loc;     	// uniform location for texture 0
		
		public JOGLCanvas(GLCapabilities caps) 
		{			
			super(caps);
			
			System.out.println("Setup JOGL Canvas");
			
			this.addGLEventListener(this);
			
			this.addMouseListener(new MouseAdapter() 
			{	@Override
				public void mousePressed(MouseEvent e) 
				{	
					//System.out.println("Mouse pressed X:" + e.getX() + ", Y:" + e.getY());
					_mouseDown = true;
					_startX = e.getX();
					_startY = e.getY();
						
					if (e.getButton()==MouseEvent.BUTTON3)
						_polygonMode = GL2GL3.GL_LINE;
					
					update(null);					
				}
				@Override
				public void mouseReleased(MouseEvent e) 
				{	
					//System.out.println("Mouse released X:" + e.getX() + ", Y:" + e.getY());
					_mouseDown = false;
					_rotX += _deltaX;
					_rotY += _deltaY;
					_deltaX = 0;
					_deltaY = 0;
										
					if (e.getButton()==MouseEvent.BUTTON3)
						_polygonMode = GL2GL3.GL_FILL;
					
					update(null);
				}
				
			});
			
			this.addMouseMotionListener(new MouseAdapter()
			{	@Override
				public void mouseDragged(MouseEvent e) 
				{
					if (_mouseDown)
					{  	//System.out.println("Mouse dragged X:" + e.getX() + ", Y:" + e.getY());
						_deltaY = e.getX() - _startX;
						_deltaX = e.getY() - _startY;
						update(null);
					}
				}
			});
			
			this.addMouseWheelListener(new MouseAdapter()
			{	@Override
				public void mouseWheelMoved(MouseWheelEvent e) 
				{	//System.out.println("Mouse wheel:" + e.getWheelRotation());
					if (e.getWheelRotation() > 0) 
						_camZ += 0.1f;
					else 
						_camZ -= 0.1f;
					update(null);
				}
			});
		}
		
	    /**
        The buildSquare method creates the vertex attributes for a textured square and VBO.
        */
        private void buildSquare(GL2GL3 gl)
        {
            // create vertex array for interleaved position, normal and texCoord    
            //            Position    , Normal     , texCrd ,
            float[] v = {-1f,  0f, -1f, 0f, -1f, 0f, 0f,  0f, // Vertex 0
                          1f,  0f, -1f, 0f, -1f, 0f, 1f,  0f, // Vertex 1
                          1f,  0f,  1f, 0f, -1f, 0f, 1f,  1f, // Vertex 2
                         -1f,  0f,  1f, 0f, -1f, 0f, 0f,  1f};// Vertex 3       
            int bytesPerFloat = Float.SIZE / Byte.SIZE;
            _vboV = glUtils.buildVBO(gl, FloatBuffer.wrap(v), 
                                     4, 8, 
                                     bytesPerFloat, 
                                     GL2GL3.GL_ARRAY_BUFFER, 
                                     GL2GL3.GL_STATIC_DRAW);
                
            // create index array for 2 GL_TRIANGLES
            _numI = 6;
            int[] i = {0, 1, 2, 0, 2, 3};
                
            // create vertex buffer objects 
            int bytesPerInt = Integer.SIZE / Byte.SIZE;        
            _vboI = glUtils.buildVBO(gl, IntBuffer.wrap(i), 
                                     _numI, 1, 
                                     bytesPerInt, 
                                     GL2GL3.GL_ELEMENT_ARRAY_BUFFER, 
                                     GL2GL3.GL_STATIC_DRAW);
            
            glUtils.getGLError(gl, "buildSquare");
        }
        
		/**
		The init method initializes the global variables and builds the shader program. 
		It should be called after a window with a valid OpenGL context is present.
		*/
		@Override 
		public void init(GLAutoDrawable ad) 
		{
			System.out.println("Init");
			GL2GL3 gl = ad.getGL().getGL2GL3();
			
			
			_modelViewMatrix = new SLMat4f();
			_projectionMatrix = new SLMat4f();
			
			buildSquare(gl);

			// Set light parameters
			_globalAmbi    = new SLVec4f( 0.0f, 0.0f, 0.0f, 1.0f);
			_lightPos      = new SLVec3f( 0.0f, 0.0f, 100.0f);   
			_lightDir      = new SLVec3f( 0.0f, 0.0f,-1.0f);   
			_lightAmbient  = new SLVec4f( 0.1f, 0.1f, 0.1f, 1.0f);  
			_lightDiffuse  = new SLVec4f( 1.0f, 1.0f, 1.0f, 1.0f);
			_lightSpecular = new SLVec4f( 1.0f, 1.0f, 1.0f, 1.0f);
			_matAmbient    = new SLVec4f( 1.0f, 1.0f, 1.0f, 1.0f);    
			_matDiffuse    = new SLVec4f( 1.0f, 1.0f, 1.0f, 1.0f);    
			_matSpecular   = new SLVec4f( 1.0f, 1.0f, 1.0f, 1.0f);    
			_matEmissive   = new SLVec4f( 0.0f, 0.0f, 0.0f, 1.0f);
			_matShininess  = 100.0f; 
			
			_polygonMode = GL2GL3.GL_FILL;
   
			// backwards movement of the camera
			_camZ = -3.0f;      

			// Mouse rotation parameters
			_rotX = 0;
			_rotY = 0;
			_deltaX = _deltaY = 0;
			_mouseDown = false;
			
			// Load textures
			_textureID = glUtils.buildTexture(gl, "_data/images/textures/earth2048_C.jpg",
					                          GL2GL3.GL_LINEAR_MIPMAP_LINEAR, 
					                          GL2GL3.GL_LINEAR,
					                          GL2GL3.GL_REPEAT, 
					                          GL2GL3.GL_REPEAT);
			
			// Load, compile & link shaders
			_shaderVertID = glUtils.buildShader(gl, "_globals/oglsl/ADSTex.vert", 
			                                    GL2GL3.GL_VERTEX_SHADER);
			_shaderFragID = glUtils.buildShader(gl, "_globals/oglsl/ADSTex.frag", 
			                                    GL2GL3.GL_FRAGMENT_SHADER);
			_shaderProgID = glUtils.buildProgram(gl, _shaderVertID, _shaderFragID);
			
			// Activate the shader program
			gl.glUseProgram(_shaderProgID); 

			// Get the variable locations (identifiers) within the program
			_pLoc            = gl.glGetAttribLocation (_shaderProgID, "a_position");
			_nLoc            = gl.glGetAttribLocation (_shaderProgID, "a_normal");
			_tLoc            = gl.glGetAttribLocation (_shaderProgID, "a_texCoord");
			_mvMatrixLoc     = gl.glGetUniformLocation(_shaderProgID, "u_mvMatrix");
			_mvpMatrixLoc    = gl.glGetUniformLocation(_shaderProgID, "u_mvpMatrix");
			_nMatrixLoc      = gl.glGetUniformLocation(_shaderProgID, "u_nMatrix");
			_globalAmbiLoc   = gl.glGetUniformLocation(_shaderProgID, "u_globalAmbi");
			_lightPosVSLoc   = gl.glGetUniformLocation(_shaderProgID, "u_lightPosVS");
			_lightDirVSLoc   = gl.glGetUniformLocation(_shaderProgID, "u_lightDirVS");
			_lightAmbientLoc = gl.glGetUniformLocation(_shaderProgID, "u_lightAmbient");
			_lightDiffuseLoc = gl.glGetUniformLocation(_shaderProgID, "u_lightDiffuse");
			_lightSpecularLoc= gl.glGetUniformLocation(_shaderProgID, "u_lightSpecular");
			_matAmbientLoc   = gl.glGetUniformLocation(_shaderProgID, "u_matAmbient");
			_matDiffuseLoc   = gl.glGetUniformLocation(_shaderProgID, "u_matDiffuse");
			_matSpecularLoc  = gl.glGetUniformLocation(_shaderProgID, "u_matSpecular");
			_matEmissiveLoc  = gl.glGetUniformLocation(_shaderProgID, "u_matEmissive");
			_matShininessLoc = gl.glGetUniformLocation(_shaderProgID, "u_matShininess");
			_texture0Loc     = gl.glGetUniformLocation(_shaderProgID, "u_texture0");      

			// Set some OpenGL states
			gl.glClearColor(0.0f, 0.0f, 0.0f, 1);  // Set the background color         
			gl.glEnable(GL2GL3.GL_DEPTH_TEST);     // Enables depth test
			gl.glEnable(GL2GL3.GL_CULL_FACE);      // Enables the culling of back faces
			
			glUtils.getGLError(gl, "Init");
		}
		
		/**
		The reshape method gets called on the resize event of the window. This method
		should be called once before the display. Do everything that is dependent on
		the size and ratio of the window.
		*/
		@Override
		public void reshape(GLAutoDrawable ad, int x, int y, int width, int height) 
		{
			//System.out.println("Reshape");
			GL2GL3 gl = ad.getGL().getGL2GL3();
			
			float w = (float)width;
			float h = (float)height;
		   
			// define the projection matrix
			_projectionMatrix.perspective(45.0f, w/h, 0.01f, 10.0f);
		   
			// define the viewport
			gl.glViewport(0, 0, width, height);
			
			glUtils.getGLError(gl, "reshape");
		}
		
		/**
		The display method does all the rendering for one frame from scratch with OpenGL.
		*/
		@Override
		public void display(GLAutoDrawable ad) 
		{
			GL2GL3 gl = ad.getGL().getGL2GL3();
			
			// Clear the color & depth buffer
			gl.glClear(GL2GL3.GL_COLOR_BUFFER_BIT | GL2GL3.GL_DEPTH_BUFFER_BIT);

			// Set filled or lined polygon mode
			gl.glPolygonMode(GL2GL3.GL_FRONT_AND_BACK, _polygonMode);
			
		   // Start with identity every frame
		   _modelViewMatrix.identity();
			
		   // View transform: move the coordinate system away from the camera
		   _modelViewMatrix.translate(0, 0, _camZ);

		   // View transform: rotate the coordinate system increasingly
		   _modelViewMatrix.rotate(_rotX + _deltaX, 1f,0f,0f);
		   _modelViewMatrix.rotate(_rotY + _deltaY, 0f,1f,0f);
		   
		   // Transform light position & direction into view space
		   SLVec3f lightPosVS = _modelViewMatrix.multiply(_lightPos);
		   
		   // The light direction is not a position. We only take the rotation of the mv matrix.
		   SLMat3f viewRot    = new SLMat3f(_modelViewMatrix.mat3());
		   SLVec3f lightDirVS = viewRot.multiply(_lightDir);
		   
		   // Rotate the model
		   _modelViewMatrix.rotate(90f, -1f,0f,0f);

		   // Build the combined modelview-projection matrix
		   SLMat4f mvp = new SLMat4f(_projectionMatrix);
		   mvp.multiply(_modelViewMatrix);
		   
		   // Normal matrix
		   SLMat3f nm = new SLMat3f(_modelViewMatrix.inverseTransposed());
		   
		   // Pass the matrix uniform variables
		   gl.glUniformMatrix4fv(_mvMatrixLoc,  1, false, _modelViewMatrix.toArray(), 0);
		   gl.glUniformMatrix3fv(_nMatrixLoc,   1, false, nm.toArray(), 0);
		   gl.glUniformMatrix4fv(_mvpMatrixLoc, 1, false, mvp.toArray(), 0);
		   
		   // Pass lighting uniforms variables
		   gl.glUniform4f(_globalAmbiLoc, _globalAmbi.x, _globalAmbi.y, _globalAmbi.z, _globalAmbi.w);
		   gl.glUniform3f(_lightPosVSLoc, lightPosVS.x, lightPosVS.y, lightPosVS.z);
		   gl.glUniform3f(_lightDirVSLoc, lightDirVS.x, lightDirVS.y, lightDirVS.z);
		   gl.glUniform4f(_lightAmbientLoc, _lightAmbient.x, _lightAmbient.y, _lightAmbient.z, _lightAmbient.w);
		   gl.glUniform4f(_lightDiffuseLoc, _lightDiffuse.x, _lightDiffuse.y, _lightDiffuse.z, _lightDiffuse.w);
		   gl.glUniform4f(_lightSpecularLoc, _lightSpecular.x, _lightSpecular.y, _lightSpecular.z, _lightSpecular.w);
		   gl.glUniform4f(_matAmbientLoc, _matAmbient.x, _matAmbient.y, _matAmbient.z, _matAmbient.w); 
		   gl.glUniform4f(_matDiffuseLoc, _matDiffuse.x, _matDiffuse.y, _matDiffuse.z, _matDiffuse.w); 
		   gl.glUniform4f(_matSpecularLoc, _matSpecular.x, _matSpecular.y, _matSpecular.z, _matSpecular.w); 
		   gl.glUniform4f(_matEmissiveLoc, _matEmissive.x, _matEmissive.y, _matEmissive.z, _matEmissive.w);
		   gl.glUniform1f(_matShininessLoc, _matShininess);
		   
		   // Pass the active texture unit
		   gl.glUniform1i(_texture0Loc, 0);
		     
		   /////////////////////////////
		   // Draw sphere with 2 VBOs //
		   /////////////////////////////

		   gl.glEnableVertexAttribArray(_pLoc);
		   gl.glEnableVertexAttribArray(_nLoc);
		   gl.glEnableVertexAttribArray(_tLoc);
		   
		   // Activate VBOs
		   gl.glBindBuffer(GL2GL3.GL_ARRAY_BUFFER, _vboV[0]);
		   gl.glBindBuffer(GL2GL3.GL_ELEMENT_ARRAY_BUFFER, _vboI[0]);

		   // Activate Texture
		   gl.glBindTexture(GL2GL3.GL_TEXTURE_2D, _textureID[0]);
		   
		   // For VBO only offset instead of data pointer
		   int sizeOfFloat = Float.SIZE / Byte.SIZE;
		   int stride  = 8 * sizeOfFloat;
		   int offsetN = 3 * sizeOfFloat;
		   int offsetT = 6 * sizeOfFloat;
		   gl.glVertexAttribPointer(_pLoc, 3, GL2GL3.GL_FLOAT, false, stride, 0);
		   gl.glVertexAttribPointer(_nLoc, 3, GL2GL3.GL_FLOAT, false, stride, offsetN);
		   gl.glVertexAttribPointer(_tLoc, 2, GL2GL3.GL_FLOAT, false, stride, offsetT);
		   
		   /////////////////////////////////////////////////////////////////////////
		   // Draw cube with triangles by indexes
		   gl.glDrawElements(GL2GL3.GL_TRIANGLES, _numI, GL2GL3.GL_UNSIGNED_INT, 0);
		   /////////////////////////////////////////////////////////////////////////

		   // Deactivate buffers
		   gl.glBindBuffer(GL2GL3.GL_ARRAY_BUFFER, 0);
		   gl.glBindBuffer(GL2GL3.GL_ELEMENT_ARRAY_BUFFER, 0);

		   // Disable the vertex arrays
		   gl.glDisableVertexAttribArray(_pLoc);
		   gl.glDisableVertexAttribArray(_nLoc);
		   gl.glDisableVertexAttribArray(_tLoc);
		   
		   // Check for errors
		   glUtils.getGLError(gl, "display");
		   
		   // Fast copy the back buffer to the front buffer is done by GLCanvas
		}
		
		/**
		dispose is called when the user closes the window and can be used for proper
		deallocation of resources.
		*/
		@Override
		public void dispose(GLAutoDrawable ad) 
		{
			System.out.println("Dispose");
			GL2GL3 gl = ad.getGL().getGL2GL3();
			
			// Delete shaders & programs on GPU
			gl.glDeleteShader(_shaderVertID);
			gl.glDeleteShader(_shaderFragID);
			gl.glDeleteProgram(_shaderProgID);
		   
			// Delete arrays & buffers on GPU
			//gl.glDeleteBuffers(1, _vboV);
			//gl.glDeleteBuffers(1, _vboI);
		}
	}
}
