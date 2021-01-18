//#############################################################################
//  File:      SLGLFrameBuffer.cpp
//  Purpose:   Wrapper class around OpenGL Frame Buffer Objects (FBO)
//  Author:    Carlos Arauz
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLFrameBuffer.h>

//-----------------------------------------------------------------------------
SLuint SLGLFrameBuffer::totalBufferSize  = 0;
SLuint SLGLFrameBuffer::totalBufferCount = 0;
//-----------------------------------------------------------------------------
//! Constructor
SLGLFrameBuffer::SLGLFrameBuffer(SLbool  renderBuffer,
                                 SLsizei rboWidth,
                                 SLsizei rboHeight)
{
    _id           = 0;
    _rbo          = 0;
    _renderBuffer = renderBuffer;
    _rboWidth     = rboWidth;
    _rboHeight    = rboHeight;
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
    if (_id)
    {
        glDeleteBuffers(1, &_id);
        _id = 0;
    }

    if (_rbo)
    {
        glDeleteBuffers(1, &_rbo);
        _rbo = 0;
    }
}
//-----------------------------------------------------------------------------
//! generate the frame buffer and the render buffer if wanted
void SLGLFrameBuffer::generate()
{
    if (_id == 0)
    {
        glGenFramebuffers(1, &_id);
        bind();

        if (_renderBuffer)
        {
            glGenRenderbuffers(1, &_rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, _rbo);
            bufferStorage(_rboWidth, _rboHeight);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                                      GL_DEPTH_ATTACHMENT,
                                      GL_RENDERBUFFER,
                                      _rbo);
        }

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
    assert(_id && "No framebuffer generated");
    glBindFramebuffer(GL_FRAMEBUFFER, _id);
}
//-----------------------------------------------------------------------------
void SLGLFrameBuffer::bindRenderBuffer()
{
    assert(_rbo && "No renderbuffer generated");
    glBindRenderbuffer(GL_RENDERBUFFER, _rbo);
}
//-----------------------------------------------------------------------------
void SLGLFrameBuffer::unbind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
//-----------------------------------------------------------------------------
//! change the render buffer size at will
void SLGLFrameBuffer::bufferStorage(SLsizei width,
                                    SLsizei height)
{
    bind();
    bindRenderBuffer();
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
    assert(_id && _rbo);
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
