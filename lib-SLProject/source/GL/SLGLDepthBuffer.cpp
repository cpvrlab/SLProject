//#############################################################################
//  File:      SLGLDepthBuffer.h
//  Purpose:   Uses an OpenGL framebuffer object as a depth-buffer
//  Author:    Michael Schertenleib
//  Date:      May 2020
//  Copyright: Michael Schertenleib
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <SLGLState.h>
#include <SLGLDepthBuffer.h>

//-----------------------------------------------------------------------------
SLGLDepthBuffer::SLGLDepthBuffer(SLVec2i dimensions,
                                 SLenum  magFilter,
                                 SLenum  minFilter,
                                 SLint   wrap,
                                 SLfloat borderColor[],
                                 SLenum  target) : _dimensions(dimensions), _target(target)
{
    assert(target == GL_TEXTURE_2D || target == GL_TEXTURE_CUBE_MAP);
    SLGLState* stateGL = SLGLState::instance();

    SLint previousFrameBuffer;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previousFrameBuffer);

    // Init framebuffer.
    glGenFramebuffers(1, &_fboID);
    glBindFramebuffer(GL_FRAMEBUFFER, _fboID);

    glGenTextures(1, &_texID);
    stateGL->activeTexture(GL_TEXTURE0 + (SLuint)_texID);
    stateGL->bindTexture(target, _texID);

    // Texture parameters.
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, magFilter);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap);

    if (borderColor != nullptr)
        glTexParameterfv(target, GL_TEXTURE_BORDER_COLOR, borderColor);

    if (target == GL_TEXTURE_2D)
    {
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_DEPTH_COMPONENT,
                     _dimensions.x,
                     _dimensions.y,
                     0,
                     GL_DEPTH_COMPONENT,
                     GL_FLOAT,
                     nullptr);

        // Attach texture to framebuffer.
        glFramebufferTexture2D(GL_FRAMEBUFFER,
                               GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D,
                               _texID,
                               0);
    }
    else // target is GL_TEXTURE_CUBE_MAP
    {
        for (SLint i = 0; i < 6; ++i)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                         0,
                         GL_DEPTH_COMPONENT,
                         _dimensions.x,
                         _dimensions.y,
                         0,
                         GL_DEPTH_COMPONENT,
                         GL_FLOAT,
                         nullptr);

            // Attach texture to framebuffer.
            glFramebufferTexture2D(GL_FRAMEBUFFER,
                                   GL_DEPTH_ATTACHMENT,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   _texID,
                                   0);
        }
    }

    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    glBindFramebuffer(GL_FRAMEBUFFER, previousFrameBuffer);

#ifdef _GLDEBUG
    GET_GL_ERROR;
#endif

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "FBO failed to initialize correctly." << std::endl;
}
//-----------------------------------------------------------------------------
SLGLDepthBuffer::~SLGLDepthBuffer()
{
    glDeleteTextures(1, &_texID);
    glDeleteFramebuffers(1, &_fboID);
}
//-----------------------------------------------------------------------------
void SLGLDepthBuffer::activateAsTexture(SLuint loc)
{
    SLGLState* stateGL = SLGLState::instance();
    stateGL->activeTexture(GL_TEXTURE0 + (SLuint)_texID);
    stateGL->bindTexture(_target, _texID);
    glUniform1i(loc, _texID);

#ifdef _GLDEBUG
    GET_GL_ERROR;
#endif
}
//-----------------------------------------------------------------------------
SLfloat SLGLDepthBuffer::depth(SLint x, SLint y)
{
    SLfloat depth;
    glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);

#ifdef _GLDEBUG
    GET_GL_ERROR;
#endif

    return depth;
}
//-----------------------------------------------------------------------------
void SLGLDepthBuffer::bind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, _fboID);

#ifdef _GLDEBUG
    GET_GL_ERROR;
#endif
}
//-----------------------------------------------------------------------------
void SLGLDepthBuffer::bindFace(SLenum face)
{
    assert(_target == GL_TEXTURE_CUBE_MAP);
    assert(face >= GL_TEXTURE_CUBE_MAP_POSITIVE_X &&
           face <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, face, _texID, 0);

#ifdef _GLDEBUG
    GET_GL_ERROR;
#endif
}
//-----------------------------------------------------------------------------
