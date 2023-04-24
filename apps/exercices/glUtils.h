//#############################################################################
//  File:      glUtils.h
//  Purpose:   General OpenGL utility functions for simple OpenGL demo apps
//  Date:      July 2014
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef GLUTILS_H
#define GLUTILS_H

#include <array>
#include <string>

using std::string;

#ifdef __APPLE__
#    if defined(TARGET_OS_IOS) && (TARGET_OS_IOS == 1)
#        include <OpenGLES/ES3/gl.h>
#        include <OpenGLES/ES3/glext.h>
#    else
#        include <GL/gl3w.h>
#    endif
#elif defined(ANDROID) || defined(ANDROID_NDK)
// https://stackoverflow.com/questions/31003863/gles-3-0-including-gl2ext-h
#    include <GLES3/gl3.h>
#    include <GLES2/gl2ext.h>
#    ifndef GL_CLAMP_TO_BORDER // see #define GL_CLAMP_TO_BORDER_OES 0x812D in gl2ext.h
#        define GL_CLAMP_TO_BORDER GL_CLAMP_TO_BORDER_OES
#    endif
//#    include <GLES3/gl31.h>
//#    include <GLES3/gl3ext.h>
#elif defined(_WIN32)
#    include <GL/gl3w.h>
#elif defined(linux) || defined(__linux) || defined(__linux__)
#    include <GL/gl3w.h>
#else
#    error "SL has not been ported to this OS"
#endif

using std::string;
using std::vector;

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

    //! Builds an GLSL-Shader program for transform feedback and returns the program id
    static GLuint buildProgramTF(GLuint vertShaderID, GLuint fragShaderID);

    //! Builds an GLSL-Shader program and returns the program id
    static GLuint buildProgram(GLuint vertShaderID,
                               GLuint geomShaderID,
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
    static GLuint build3DTexture(const vector<string>&         files,
                                 GLuint&                       x_extend,
                                 GLuint&                       y_extend,
                                 GLuint&                       z_extend,
                                 GLint                         min_filter  = GL_LINEAR,
                                 GLint                         mag_filter  = GL_LINEAR,
                                 GLint                         wrapR       = GL_CLAMP_TO_BORDER,
                                 GLint                         wrapS       = GL_CLAMP_TO_BORDER,
                                 GLint                         wrapT       = GL_CLAMP_TO_BORDER,
                                 const std::array<GLfloat, 4>& borderColor = {0.0f, 0.0f, 0.0f, 0.0f});

    //! Checks if an OpenGL error occurred
    static void getGLError(const char* file, int line, bool quit);

    //! Returns the GLSL version string
    static string glSLVersionNO();
};
//-----------------------------------------------------------------------------
#endif
