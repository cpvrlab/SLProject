//#############################################################################
//  File:      SLGLDepthBuffer.h
//  Purpose:   Uses an OpenGL framebuffer object as a depth-buffer
//  Author:    Michael Schertenleib
//  Date:      May 2020
//  Copyright: Michael Schertenleib
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLDEPTHBUFFER_H
#define SLGLDEPTHBUFFER_H

#include <SLGLState.h>
#include <SL.h>

//-----------------------------------------------------------------------------
class SLGLDepthBuffer
{
public:
    SLGLDepthBuffer(SLVec2i dimensions,
                    SLenum  magFilter     = GL_NEAREST,
                    SLenum  minFilter     = GL_NEAREST,
                    SLint   wrap          = GL_REPEAT,
                    SLfloat borderColor[] = nullptr,
                    SLenum  target        = GL_TEXTURE_2D);
    ~SLGLDepthBuffer();

    SLint    texID() { return _texID; }
    SLint    target() { return _target; }
    void     activateAsTexture(SLuint loc);
    void     bind();
    void     unbind();
    void     bindFace(SLenum face);
    SLfloat* readPixels();
    SLVec2i  dimensions() { return _dimensions; }

private:
    SLVec2i _dimensions; //<! Size of the texture
    SLuint  _fboID;      //<! ID of the framebuffer object
    SLuint  _texID;      //<! ID of the texture
    SLenum  _target;     //<! GL_TEXTURE_2D or GL_TEXTURE_CUBE_MAP
};
//-----------------------------------------------------------------------------
#endif //SLGLDEPTHBUFFER_H
