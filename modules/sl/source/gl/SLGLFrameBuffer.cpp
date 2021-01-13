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
    this->_id           = 0;
    this->_rbo          = 0;
    this->_renderBuffer = renderBuffer;
    this->_rboWidth     = rboWidth;
    this->_rboHeight    = rboHeight;
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
    if (this->_id == 0)
    {
        glGenFramebuffers(1, &this->_id);
        glBindFramebuffer(GL_FRAMEBUFFER, this->_id);

        if (this->_renderBuffer)
        {
            glGenRenderbuffers(1, &this->_rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, this->_rbo);
            this->bufferStorage(this->_rboWidth, this->_rboHeight);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, this->_rbo);
        }

        // test if the generated fbo is valid
        if ((glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE)
            SL_EXIT_MSG("Frame buffer creation failed!");

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        GET_GL_ERROR;

        totalBufferCount++;
    }
}
//-----------------------------------------------------------------------------
void SLGLFrameBuffer::bind()
{
    assert(this->_id && "No framebuffer generated");
    glBindFramebuffer(GL_FRAMEBUFFER, this->_id);
}
//-----------------------------------------------------------------------------
void SLGLFrameBuffer::bindRenderBuffer()
{
    assert(this->_rbo && "No renderbuffer generated");
    glBindRenderbuffer(GL_RENDERBUFFER, this->_rbo);
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
    this->bind();
    this->bindRenderBuffer();
    this->_rboWidth  = width;
    this->_rboHeight = height;
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
    assert(this->_id && this->_rbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           attachment,
                           target,
                           texture->texID(),
                           level);
}
//-----------------------------------------------------------------------------
