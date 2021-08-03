#include "SENSGLTextureReader.h"

#include <SENS.h>
#include <Utils.h>

//include opengl plattform dependent
#if defined(SENS_OS_MACIOS)
#    include <OpenGLES/ES3/gl.h>
#    include <OpenGLES/ES3/glext.h>
#elif defined(SENS_OS_MACOS)
#    include <GL/gl3w.h>
#elif defined(SENS_OS_ANDROID)
//https://stackoverflow.com/questions/31003863/gles-3-0-including-gl2ext-h
#    include <GLES3/gl3.h>
#    include <GLES2/gl2ext.h>
#    ifndef GL_CLAMP_TO_BORDER //see #define GL_CLAMP_TO_BORDER_OES 0x812D in gl2ext.h
#        define GL_CLAMP_TO_BORDER GL_CLAMP_TO_BORDER_OES
#    endif
#elif defined(SENS_OS_WINDOWS)
#    include <GL/gl3w.h>
#elif defined(SENS_OS_LINUX)
#    include <GL/gl3w.h>
#else
#    error "SL has not been ported to this OS"
#endif

#ifndef GET_GL_ERROR
#    if defined(DEBUG) || defined(_DEBUG)
#        define GET_GL_ERROR SENSGLTextureReader::getGLError((const char*)__FILE__, __LINE__, false)
#    else
#        define GET_GL_ERROR
#    endif
#endif

//-----------------------------------------------------------------------------
std::vector<std::string> SENSGLTextureReader::_errors;

//-----------------------------------------------------------------------------
//! Returns the OpenGL Shading Language version number as a string.
/*! The string returned by glGetString can contain additional vendor
 information such as the build number and the brand name.
 For the shading language string "Nvidia GLSL 4.5" the function returns "450"
 */
std::string glSLVersionNO()
{
    std::string versionStr = std::string((const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
    size_t      dotPos     = versionStr.find('.');
    char        NO[4];
    NO[0] = versionStr[dotPos - 1];
    NO[1] = versionStr[dotPos + 1];
    NO[2] = '0';
    NO[3] = 0;
    return std::string(NO);
}

//-----------------------------------------------------------------------------
GLuint buildShaderFromSource(std::string source, GLenum shaderType, bool isGlExternalTexture)
{
    // Compile Shader code
    GLuint      shaderHandle = glCreateShader(shaderType);
    std::string version;

    std::string versionGLSL = glSLVersionNO();
    std::string glVersion   = std::string((const char*)glGetString(GL_VERSION));
    bool        glIsES3     = (glVersion.find("OpenGL ES 3") != string::npos);
    std::string srcVersion  = "#version " + versionGLSL;
    if (glIsES3)
        srcVersion += " es";
    srcVersion += "\n";

    std::string completeSrc = srcVersion + source;

    if (isGlExternalTexture)
    {
        Utils::replaceString(completeSrc, "#include extension", "#extension GL_OES_EGL_image_external_essl3 : enable");
        Utils::replaceString(completeSrc, "#include sampler", "uniform samplerExternalOES");
    }
    else
    {
        Utils::replaceString(completeSrc, "#include extension", "");
        Utils::replaceString(completeSrc, "#include sampler", "uniform sampler2D");
    }

    const char* src = completeSrc.c_str();

    glShaderSource(shaderHandle, 1, &src, nullptr);
    glCompileShader(shaderHandle);

    // Check compile success
    GLint compileSuccess;
    glGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &compileSuccess);

    if (!compileSuccess)
    {
        GLint logSize = 0;
        glGetShaderiv(shaderHandle, GL_INFO_LOG_LENGTH, &logSize);

        GLchar* log = new GLchar[logSize];

        glGetShaderInfoLog(shaderHandle, logSize, nullptr, log);

        Utils::log("Application", "Cannot compile shader %s", log);
        Utils::log("Application", "%s", src);
        exit(1);
    }
    return shaderHandle;
}

SENSGLTextureReader::SENSGLTextureReader(unsigned int textureId, bool isGlTextureExternal, int targetWidth, int targetHeight)
  : _extTextureId(textureId),
    _isGlTextureExternal(isGlTextureExternal),
    _targetWidth(targetWidth),
    _targetHeight(targetHeight)
{
    initGl();
}

SENSGLTextureReader::~SENSGLTextureReader()
{
    glDeleteTextures(1, &_targetTex);
    glDeleteFramebuffers(1, &_fbo);
    glDeleteVertexArrays(1, &_VAO);
    glDeleteBuffers(1, &_VBO);
    glDeleteBuffers(1, &_EBO);
    glDeleteProgram(_prog);
}

void SENSGLTextureReader::initGl()
{
    //-----------------------------------------------------------------------------
    //store old gl state
    GLint lastFBO = -1;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &lastFBO);
    GLint lastTex = -1;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &lastTex);

    //-----------------------------------------------------------------------------
    //setup shader program
    static std::string vertexShSrc =
      "#ifdef GL_ES\n"
      "   precision highp float;\n"
      "#endif\n"
      "\n"
      "layout (location = 0) in vec2 aPos;\n"
      "layout (location = 1) in vec2 aTexCoords;\n"
      "\n"
      "out vec2 TexCoords;\n"
      "\n"
      "void main()\n"
      "{\n"
      "    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);\n"
      "    TexCoords = aTexCoords;\n"
      "}\n";

    static std::string fragShSrc =
      "#include extension\n"
      "#ifdef GL_ES\n"
      "   precision highp float;\n"
      "#endif\n"
      "\n"
      "out vec4 FragColor;\n"
      "\n"
      "in vec2 TexCoords;\n"
      "\n"
      "#include sampler texture0;\n"
      "\n"
      "void main()\n"
      "{\n"
      "    //ATTENTION: order is changed to bgr\n"
      "    FragColor = texture(texture0, vec2(TexCoords.x, TexCoords.y)).bgra;\n"
      "}\n";

    GLuint vertexSh = buildShaderFromSource(vertexShSrc, GL_VERTEX_SHADER, _isGlTextureExternal);
    GLuint fragSh   = buildShaderFromSource(fragShSrc, GL_FRAGMENT_SHADER, _isGlTextureExternal);
    _prog           = glCreateProgram();
    glAttachShader(_prog, vertexSh);
    glAttachShader(_prog, fragSh);
    glLinkProgram(_prog);
    glDeleteShader(vertexSh);
    glDeleteShader(fragSh);
    GET_GL_ERROR;

    // clang-format off
    //-----------------------------------------------------------------------------
    //setup vertex array
    float vertices[] = {
        // positions          // texture coords
         1.0f,  1.0f,   1.0f, 1.0f,   // top right
         1.0f, -1.0f,   1.0f, 0.0f,   // bottom right
        -1.0f, -1.0f,   0.0f, 0.0f,   // bottom left
        -1.0f,  1.0f,   0.0f, 1.0f    // top left
    };
    //ATTENTION: BE SURE ORDER INDICATES A FRONT FACING NORMAL
    unsigned int indices[] = {
        0, 3, 1, // first triangle
        1, 3, 2  // second triangle
    };
    // clang-format on

    glGenVertexArrays(1, &_VAO);
    glGenBuffers(1, &_VBO);
    glGenBuffers(1, &_EBO);

    glBindVertexArray(_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glGenFramebuffers(1, &_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    GET_GL_ERROR;

    //-----------------------------------------------------------------------------
    //setup target texture
    glGenTextures(1, &_targetTex);
    glBindTexture(GL_TEXTURE_2D, _targetTex);
    //TODO: GL_RGB??
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _targetWidth, _targetHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //bind fbo to target texture
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _targetTex, 0);
    GET_GL_ERROR;

    //test fbo status
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
        Utils::log("SENSGLTextureReader", "failed to make complete framebuffer object %x", status);
    GET_GL_ERROR;
    //-----------------------------------------------------------------------------
    //restore old gl state
    glBindFramebuffer(GL_FRAMEBUFFER, lastFBO);
    glBindTexture(GL_TEXTURE_2D, lastTex);
}

cv::Mat SENSGLTextureReader::readImageFromGpu()
{
    //-----------------------------------------------------------------------------
    //store old gl state
    GLint lastFBO = -1;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &lastFBO);
    GLint lastTex = -1;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &lastTex);
    //depth test
    GLboolean lastDepthTestV;
    glGetBooleanv(GL_DEPTH_TEST, &lastDepthTestV);
    //stencil test
    GLboolean lastStencilTestV;
    glGetBooleanv(GL_STENCIL_TEST, &lastStencilTestV);
    //viewport
    GLint lastViewport[4];
    glGetIntegerv(GL_VIEWPORT, lastViewport);

    //-----------------------------------------------------------------------------
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    glViewport(0, 0, _targetWidth, _targetHeight);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    if (lastDepthTestV)
        glDisable(GL_DEPTH_TEST);
    if (lastStencilTestV)
        glDisable(GL_STENCIL_TEST);

    glBindTexture(GL_TEXTURE_2D, _extTextureId);

    glUseProgram(_prog);
    glBindVertexArray(_VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    //read pixels from framebuffer
    cv::Mat image = cv::Mat(_targetHeight, _targetWidth, CV_8UC4);
    glReadPixels(0, 0, _targetWidth, _targetHeight, GL_RGBA, GL_UNSIGNED_BYTE, image.data);

    //-------------------------------------------------------------------
    //restore old gl state
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, lastFBO);
    glBindTexture(GL_TEXTURE_2D, lastTex);
    glViewport(lastViewport[0], lastViewport[1], lastViewport[2], lastViewport[3]);
    if (lastDepthTestV)
        glEnable(GL_DEPTH_TEST);
    if (lastStencilTestV)
        glEnable(GL_STENCIL_TEST);
    GET_GL_ERROR;

    if (image.data)
        return image;
    else
        return cv::Mat();
}

//-----------------------------------------------------------------------------
void SENSGLTextureReader::getGLError(const char* file,
                                     int         line,
                                     bool        quit)
{
    GLenum err;
    if ((err = glGetError()) != GL_NO_ERROR)
    {
        std::string errStr;
        switch (err)
        {
            case GL_INVALID_ENUM:
                errStr = "GL_INVALID_ENUM";
                break;
            case GL_INVALID_VALUE:
                errStr = "GL_INVALID_VALUE";
                break;
            case GL_INVALID_OPERATION:
                errStr = "GL_INVALID_OPERATION";
                break;
            case GL_INVALID_FRAMEBUFFER_OPERATION:
                errStr = "GL_INVALID_FRAMEBUFFER_OPERATION";
                break;
            case GL_OUT_OF_MEMORY:
                errStr = "GL_OUT_OF_MEMORY";
                break;
            default:
                errStr = "Unknown error";
        }

        // Build error string as a concatenation of file, line & error
        char sLine[32];
        sprintf(sLine, "%d", line);

        std::string newErr(file);
        newErr += ": line:";
        newErr += sLine;
        newErr += ": ";
        newErr += errStr;

        // Check if error exists already
        bool errExists = std::find(_errors.begin(),
                                   _errors.end(),
                                   newErr) != _errors.end();
        // Only print
        if (!errExists)
        {
            _errors.push_back(newErr);
            Utils::log("SENSGLTextureReader", "OpenGL Error in %s, line %d: %s\n", file, line, errStr.c_str());
        }

        if (quit)
            exit(1);
    }
}
