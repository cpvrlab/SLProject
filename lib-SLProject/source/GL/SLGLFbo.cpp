//#############################################################################
//  File:      SLGLFbo.cpp
//  Purpose:   Wraps an OpenGL framebuffer object
//  Author:    Stefan Thöni
//  Date:      September 2018
//  Copyright: Stefan Thöni
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLFbo.h>

#include <iostream>

//-----------------------------------------------------------------------------
SLGLFbo::SLGLFbo(GLuint w,
                 GLuint h,
                 GLenum magFilter,
                 GLenum minFilter,
                 GLint  internalFormat,
                 GLint  format,
                 GLint  wrap) : width(w), height(h)
{
    GLint previousFrameBuffer;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previousFrameBuffer);

    // Init framebuffer.
    glGenFramebuffers(1, &frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

    glGenTextures(1, &textureColorBuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorBuffer);

    // Texture parameters.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 internalFormat,
                 w,
                 h,
                 0,
                 GL_RGBA,
                 format,
                 nullptr);

    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D,
                           textureColorBuffer,
                           0);

    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);

    // Use a single rbo for both depth and stencil buffer
    glRenderbufferStorage(GL_RENDERBUFFER,
                          GL_DEPTH_COMPONENT24,
                          w,
                          h);

    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                              GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER,
                              rbo);

    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER,
                      previousFrameBuffer);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "FBO failed to initialize correctly." << std::endl;
}
//-----------------------------------------------------------------------------
SLGLFbo::~SLGLFbo()
{
    glDeleteTextures(1, &textureColorBuffer);
    glDeleteFramebuffers(1, &frameBuffer);
}
//-----------------------------------------------------------------------------
void SLGLFbo::activateAsTexture(const int progId,
                                const SLstring&  glSamplerName,
                                const int textureUnit)
{
    glActiveTexture(GL_TEXTURE0 + textureUnit);
    glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
    glUniform1i(glGetUniformLocation(progId,
                                     glSamplerName.c_str()),
                textureUnit);
}
//-----------------------------------------------------------------------------
