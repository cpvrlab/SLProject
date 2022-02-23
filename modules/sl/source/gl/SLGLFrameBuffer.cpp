//#############################################################################
//  File:      SLGLFrameBuffer.cpp
//  Purpose:   Wrapper class around OpenGL Frame Buffer Objects (FBO)
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Authors:   Carlos Arauz, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLFrameBuffer.h>

//-----------------------------------------------------------------------------
SLuint SLGLFrameBuffer::totalBufferSize  = 0;
SLuint SLGLFrameBuffer::totalBufferCount = 0;
//-----------------------------------------------------------------------------
//! Constructor
SLGLFrameBuffer::SLGLFrameBuffer(SLsizei rboWidth,
                                 SLsizei rboHeight)
{
    _fboId     = 0;
    _rboId     = 0;
    _prevFboId = 0;
    _rboWidth  = rboWidth;
    _rboHeight = rboHeight;
}
//-----------------------------------------------------------------------------
//! clear delete buffers and respectively adjust the stats variables
void SLGLFrameBuffer::clear()
{
    deleteGL();
    totalBufferCount--;
    totalBufferSize -= _sizeBytes;
}
//-----------------------------------------------------------------------------
//! calls the delete functions only if the buffers exist
void SLGLFrameBuffer::deleteGL()
{
    unbind();

    if (_fboId)
    {
        glDeleteBuffers(1, &_fboId);
        _fboId = 0;
    }

    if (_rboId)
    {
        glDeleteBuffers(1, &_rboId);
        _rboId = 0;
    }
}
//-----------------------------------------------------------------------------
//! generate the frame buffer and the render buffer if wanted
void SLGLFrameBuffer::generate()
{
    if (_fboId == 0)
    {
        glGenFramebuffers(1, &_fboId);
        glGenRenderbuffers(1, &_rboId);
        bindAndSetBufferStorage(_rboWidth, _rboHeight);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                                  GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER,
                                  _rboId);

        // test if the generated fbo is valid
        if ((glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE)
            SL_EXIT_MSG("Frame buffer creation failed!");

        unbind();
        GET_GL_ERROR;

        totalBufferCount++;
    }
}
//-----------------------------------------------------------------------------
void SLGLFrameBuffer::bind()
{
    assert(_fboId && "No framebuffer generated");

    // Keep the previous FB ID for later unbinding
    SLint prevFboId;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFboId);
    if (prevFboId != _fboId)
        _prevFboId = prevFboId;

    glBindFramebuffer(GL_FRAMEBUFFER, _fboId);

    assert(_rboId && "No renderbuffer generated");
    glBindRenderbuffer(GL_RENDERBUFFER, _rboId);
}
//-----------------------------------------------------------------------------
void SLGLFrameBuffer::unbind()
{
    // iOS does not allow binding to 0. That's why we keep the previous FB ID
    glBindFramebuffer(GL_FRAMEBUFFER, _prevFboId);

    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}
//-----------------------------------------------------------------------------
//! change the render buffer size at will
void SLGLFrameBuffer::bindAndSetBufferStorage(SLsizei width,
                                              SLsizei height)
{
    bind();
    _rboWidth  = width;
    _rboHeight = height;
    glRenderbufferStorage(GL_RENDERBUFFER,
                          GL_DEPTH_COMPONENT24,
                          width,
                          height);
}
//-----------------------------------------------------------------------------
//! attach one 2D texture to the frame buffer
void SLGLFrameBuffer::attachTexture2D(SLenum       attachment,
                                      SLenum       target,
                                      SLGLTexture* texture,
                                      SLint        level)
{
    assert(_fboId && _rboId);
    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           attachment,
                           target,
                           texture->texID(),
                           level);
}
//-----------------------------------------------------------------------------
/*
SLfloat* SLGLFrameBuffer::readPixels() const
{
    SLfloat* depth = new SLfloat[_dimensions.y * _dimensions.x];
    glReadPixels(0,
                 0,
                 _rboWidth,
                 _rboHeight,
                 GL_DEPTH_COMPONENT,
                 GL_FLOAT,
                 depth);
    GET_GL_ERROR;
    return depth;
}
 */
//-----------------------------------------------------------------------------
