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
#include <Instrumentor.h>

//-----------------------------------------------------------------------------
SLGLDepthBuffer::SLGLDepthBuffer(SLVec2i dimensions,
                                 SLenum  magFilter,
                                 SLenum  minFilter,
                                 SLint   wrap,
                                 SLfloat borderColor[],
                                 SLenum  target) : _dimensions(dimensions), _target(target)
{
    PROFILE_FUNCTION();

    assert(target == GL_TEXTURE_2D || target == GL_TEXTURE_CUBE_MAP);
    SLGLState* stateGL = SLGLState::instance();

    SLint previousFrameBuffer;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previousFrameBuffer);

    // Init framebuffer.
    glGenFramebuffers(1, &_fboID);
    GET_GL_ERROR;

    glBindFramebuffer(GL_FRAMEBUFFER, _fboID);
    GET_GL_ERROR;

    glGenTextures(1, &_texID);
    stateGL->activeTexture(GL_TEXTURE0 + (SLuint)_texID);
    stateGL->bindTexture(target, _texID);
    GET_GL_ERROR;

    // Texture parameters.
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, magFilter);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap);

#ifndef SL_GLES
    if (borderColor != nullptr)
        glTexParameterfv(target, GL_TEXTURE_BORDER_COLOR, borderColor);
#endif

    GET_GL_ERROR;

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
        GET_GL_ERROR;

        // Attach texture to framebuffer.
        glFramebufferTexture2D(GL_FRAMEBUFFER,
                               GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D,
                               _texID,
                               0);
        GET_GL_ERROR;
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
            GET_GL_ERROR;

            // Attach texture to framebuffer.
            glFramebufferTexture2D(GL_FRAMEBUFFER,
                                   GL_DEPTH_ATTACHMENT,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   _texID,
                                   0);
            GET_GL_ERROR;
        }
    }

#ifndef SL_GLES
    glDrawBuffer(GL_NONE);
    GET_GL_ERROR;
#endif

    glReadBuffer(GL_NONE);
    GET_GL_ERROR;

    glBindFramebuffer(GL_FRAMEBUFFER, previousFrameBuffer);
    GET_GL_ERROR;

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        SL_LOG("FBO failed to initialize correctly.");
}
//-----------------------------------------------------------------------------
SLGLDepthBuffer::~SLGLDepthBuffer()
{
    glDeleteTextures(1, &_texID);
    GET_GL_ERROR;

    glDeleteFramebuffers(1, &_fboID);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
void SLGLDepthBuffer::activateAsTexture(SLuint loc)
{
    SLGLState* stateGL = SLGLState::instance();
    stateGL->activeTexture(GL_TEXTURE0 + (SLuint)_texID);
    stateGL->bindTexture(_target, _texID);
    glUniform1i(loc, _texID);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
SLfloat* SLGLDepthBuffer::readPixels()
{
    SLfloat *depth = new SLfloat[sizeof(SLfloat) * _dimensions.y * _dimensions.x];
    glReadPixels(0, 0, _dimensions.x, _dimensions.y, GL_DEPTH_COMPONENT, GL_FLOAT, depth);
    GET_GL_ERROR;
    return depth;
}
//-----------------------------------------------------------------------------
void SLGLDepthBuffer::bind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, _fboID);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
void SLGLDepthBuffer::unbind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
void SLGLDepthBuffer::bindFace(SLenum face)
{
    assert(_target == GL_TEXTURE_CUBE_MAP);
    assert(face >= GL_TEXTURE_CUBE_MAP_POSITIVE_X &&
           face <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, face, _texID, 0);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
