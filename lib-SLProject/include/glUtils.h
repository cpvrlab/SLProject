//#############################################################################
//  File:      glUtils.h
//  Purpose:   General OpenGL utility functions for simple OpenGL demo apps
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef GLUTILS_H
#define GLUTILS_H

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <array>

using namespace std;

//-----------------------------------------------------------------------------
#define GETGLERROR glUtils::getGLError((const char*)__FILE__, __LINE__, false)
//-----------------------------------------------------------------------------
//! OpenGL utility class with functions for simple OpenGL demo apps
class glUtils
{
    public:
    static void printGLInfo();

    //! Loads an GLSL-shader file and returns the code as a string
    static string loadShader(const string& filename);

    static GLuint buildShaderFromSource(string source,
                                        GLenum shaderType,
                                        bool&  return_value);

    //! Builds an GLSL-Shader object and returns the shader id
    static GLuint buildShader(const string& shaderFile,
                              GLenum        shaderType);

    //! Builds an GLSL-Shader program and returns the program id
    static GLuint buildProgram(GLuint vertShaderID,
                               GLuint fragShaderID);

    //! Builds an OpenGL Vertex Buffer Object
    static void buildVBO(GLuint& vboID,
                         void*   dataPointer,
                         GLint   numElements,
                         GLint   elementSizeBytes,
                         GLuint  targetTypeGL = GL_ARRAY_BUFFER,
                         GLuint  usageTypeGL  = GL_STATIC_DRAW);

    //! Builds an OpenGL Vertex Array Object
    static void buildVAO(GLuint& vaoID,
                         GLuint& vboIDVertices,
                         GLuint& vboIDIndices,
                         void*   dataPointerVertices,
                         GLint   numVertices,
                         GLint   sizeVertexBytes,
                         void*   dataPointerIndices,
                         GLint   numIndices,
                         GLuint  sizeIndexBytes,
                         GLint   shaderProgramID,
                         GLint   attributePositionLoc,
                         GLint   attributeColorLoc    = -1,
                         GLint   attributeNormalLoc   = -1,
                         GLint   attributeTexCoordLoc = -1);

    //! Builds an OpenGL texture and returns the texture id
    static GLuint buildTexture(string textureFile,
                               GLint  min_filter = GL_LINEAR_MIPMAP_LINEAR,
                               GLint  mag_filter = GL_LINEAR,
                               GLint  wrapS      = GL_REPEAT,
                               GLint  wrapT      = GL_REPEAT);

    // ! Builds an OpenGL 3D texture and returns the texture id
    static GLuint build3DTexture(const vector<string>&    files,
                                 GLuint&                  x_extend,
                                 GLuint&                  y_extend,
                                 GLuint&                  z_extend,
                                 GLint                    min_filter  = GL_LINEAR,
                                 GLint                    mag_filter  = GL_LINEAR,
                                 GLint                    wrapR       = GL_CLAMP_TO_BORDER,
                                 GLint                    wrapS       = GL_CLAMP_TO_BORDER,
                                 GLint                    wrapT       = GL_CLAMP_TO_BORDER,
                                 const array<GLfloat, 4>& borderColor = {0.0f, 0.0f, 0.0f, 0.0f});

    //! Checks if an OpenGL error occurred
    static void getGLError(const char* file, int line, bool quit);

    //! Returns the GLSL version string
    static SLstring glSLVersionNO();
};
//-----------------------------------------------------------------------------
#endif
