//#############################################################################
//  File:      SLGLFrameBuffer.h
//  Purpose:   Wrapper class around OpenGL Frame Buffer Objects (FBO)
//  Author:    Carlos Arauz
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLFRAMEBUFFER_H
#define SLGLFRAMEBUFFER_H

#include <SLGLTexture.h>

//-----------------------------------------------------------------------------
/*!
The frame buffer class generates a frame buffer and a render buffer, with the
default size of 512x512, this can also in run time be changed.
*/
class SLGLFrameBuffer
{
public:
    SLGLFrameBuffer(SLbool  renderBuffer = false,
                    SLsizei rboWidth     = 512,
                    SLsizei rboHeight    = 512);
    virtual ~SLGLFrameBuffer() { clear(); }

    //! Calls delete and clears data
    void clear();

    //! Deletes this buffers
    void deleteGL();

    //! Generates the framebuffer
    void generate();

    //! Binds the framebuffer
    void bind();

    //! Binds the renderbuffer
    void bindRenderBuffer();

    //! Unbinds the framebuffer
    void unbind();

    //! Sets the size of the buffer storage
    void bufferStorage(SLsizei width,
                       SLsizei height);

    //! Attaches texture image to framebuffer
    void attachTexture2D(SLenum       attachment,
                         SLenum       target,
                         SLGLTexture* texture,
                         SLint        level = 0);

    // Getters
    SLuint  fboId() { return this->_fboId; }
    SLuint  rboId() { return this->_rboId; }
    SLsizei rboWidth() { return this->_rboWidth; }
    SLsizei rboHeight() { return this->_rboHeight; }

    // Some statistics
    static SLuint totalBufferCount; //! static total no. of buffers in use
    static SLuint totalBufferSize;  //! static total size of all buffers in bytes

protected:
    SLuint  _fboId;        //!< frame buffer identifier
    SLuint  _prevFboId;    //!< previously active frame buffer identifier
    SLuint  _rboId;        //!< render buffer identifier
    SLuint  _sizeBytes;    //!< size in bytes of this buffer
    SLsizei _rboWidth;     //!< width of the render buffer, default: 512
    SLsizei _rboHeight;    //!< height of the render buffer, default: 512
    SLbool  _renderBuffer; //!< should a renderbuffer be created?
};
//-----------------------------------------------------------------------------
#endif
