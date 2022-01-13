//#############################################################################
//  File:      SLGLFbo.h
//  Purpose:   Wraps an OpenGL framebuffer object
//  Date:      September 2018
//  Authors:   Stefan Thoeni
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLFBO_H
#define SLGLFBO_H

#include <SL.h>

//-----------------------------------------------------------------------------
class SLGLFbo
{
public:
    SLGLFbo(SLuint w,
            SLuint h,
            SLenum magFilter      = GL_NEAREST,
            SLenum minFilter      = GL_NEAREST,
            SLint  internalFormat = GL_RGB16F,
            SLint  format         = GL_FLOAT,
            SLint  wrap           = GL_REPEAT);

    ~SLGLFbo();

    void activateAsTexture(int             progId,
                           const SLstring& samplerName,
                           int             textureUnit = 0);

    SLuint width;
    SLuint height;
    SLuint attachment;
    SLuint fboID;
    SLuint texID;
    SLuint rboID;
};
//-----------------------------------------------------------------------------
#endif // SLGLFBO_H