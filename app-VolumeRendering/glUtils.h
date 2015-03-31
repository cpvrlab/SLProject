//#############################################################################
//  File:      Globals/GL/glUtils.h
//  Purpose:   General OpenGL utility functions for simple OpenGL demo apps
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef GLUTILS_H
#define GLUTILS_H

#include <stdafx.h>           // precompiled headers
#include <SLGLBuffer.h>
#include <SLCamera.h>

#include <array>

using namespace std;

#define GET_GL_ERROR glUtils::getGLError((SLchar*)__FILE__, __LINE__, false)

//-----------------------------------------------------------------------------
//! OpenGL utility class with functions for simple OpenGL demo apps
class glUtils
{
    public: 
		static void printGLInfo();

        //! Loads an GLSL-shader file and returns the code as a string
        static string loadShader  (string filename);

        //! Builds an GLSL-Shader object and returns the shader id
        static GLuint buildShader (string shaderFile, 
                                   GLenum shaderType);
   
        //! Builds an GLSL-Shader program and returns the program id
        static GLuint buildProgram(GLuint vertShaderID, 
                                    GLuint fragShaderID);

        //! Builds a vertex buffer object and returns the vbo id
        static GLuint buildVBO    (void*   dataPointer, 
                                   GLint   numElements, 
                                   GLint   elementSize, 
                                   GLuint  typeSize,        
                                   GLuint  targetTypeGL = GL_ARRAY_BUFFER,
                                   GLuint  usageTypeGL = GL_STATIC_DRAW);
   
        //! Builds an OpenGL texture and returns the texture id
        static GLuint buildTexture(string textureFile,
                                   GLint min_filter = GL_LINEAR_MIPMAP_LINEAR,
                                   GLint mag_filter = GL_LINEAR,
                                   GLint wrapS = GL_REPEAT, 
                                   GLint wrapT = GL_REPEAT);

        // ! Builds an OpenGL 3D texture and returns the texture id
        static GLuint build3DTexture(const std::vector<std::string> &files, int &x_extend, int &y_extend, int &z_extend,
                                     GLint min_filter = GL_LINEAR,
                                     GLint mag_filter = GL_LINEAR,
                                     GLint wrapR = GL_CLAMP_TO_BORDER,
                                     GLint wrapS = GL_CLAMP_TO_BORDER,
                                     GLint wrapT = GL_CLAMP_TO_BORDER,
									 const std::array<GLfloat, 4> &borderColor = { 0.0f, 0.0f, 0.0f, 0.0f }
                                     );
        //! Checks if an OpenGL error occurred
        static void getGLError(char* file, int line, bool quit);

        //! Returns a vector of sorted file names within a directory
        static SLVstring getFileNamesInDir(SLstring dirName);
};
//-----------------------------------------------------------------------------
#endif
