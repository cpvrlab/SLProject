//#############################################################################
//  File:      SLGLDepthBuffer.h
//  Purpose:   Uses an OpenGL framebuffer object as a depth-buffer
//  Date:      May 2020
//  Authors:   Michael Schertenleib
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLDEPTHBUFFER_H
#define SLGLDEPTHBUFFER_H

#include <SLObject.h>
#include <SLGLState.h>
#include <SL.h>

//-----------------------------------------------------------------------------
class SLGLDepthBuffer : SLObject
{
public:
    SLGLDepthBuffer(const SLVec2i& dimensions,
                    SLenum         magFilter     = GL_NEAREST,
                    SLenum         minFilter     = GL_NEAREST,
                    SLint          wrap          = GL_REPEAT,
                    SLfloat        borderColor[] = nullptr,
                    SLenum         target        = GL_TEXTURE_2D,
                    SLstring       name          = "SM-DepthBuffer");
    ~SLGLDepthBuffer();

    SLint    texID() { return _texID; }
    SLint    target() { return _target; }
    void     bindActive(SLuint texUnit) const;
    void     bind();
    void     unbind();
    void     bindFace(SLenum face) const;
    SLfloat* readPixels() const;
    SLVec2i  dimensions() { return _dimensions; }

private:
    SLVec2i _dimensions; //!< Size of the texture
    SLuint  _fboID;      //!< ID of the FB object
    SLint   _prevFboID;  //!< ID of the previously bound FB
    SLuint  _texID;      //!< ID of the texture
    SLenum  _target;     //!< GL_TEXTURE_2D or GL_TEXTURE_CUBE_MAP
};
//-----------------------------------------------------------------------------
typedef std::vector<SLGLDepthBuffer*> SLGLVDepthBuffer;
//-----------------------------------------------------------------------------
#endif // SLGLDEPTHBUFFER_H
