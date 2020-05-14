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
                    float   borderColor[] = nullptr);

    ~SLGLDepthBuffer();

    void    activateAsTexture(int loc);
    float   getDepth(SLint x, SLint y);
    void    bind();
    SLVec2i dimensions() { return _dimensions; }

private:
    SLVec2i _dimensions;
    SLuint  _fboID;
    SLuint  _texID;
};
//-----------------------------------------------------------------------------
#endif //SLGLDEPTHBUFFER_H
