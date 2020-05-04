//#############################################################################
//  File:      SLGLFbo.h
//  Purpose:   Wraps an OpenGL framebuffer object
//  Author:    Stefan Thoeni
//  Date:      September 2018
//  Copyright: Stefan Thoeni
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

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
                           int             textureUnit = GL_TEXTURE0);

    SLuint width;
    SLuint height;
    SLuint attachment;
    SLuint fboID;
    SLuint texID;
    SLuint rboID;
};
//-----------------------------------------------------------------------------