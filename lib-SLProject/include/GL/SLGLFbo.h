//#############################################################################
//  File:      SLGLFbo.h
//  Purpose:   Wraps an OpenGL framebuffer object
//  Author:    Stefan Thöni
//  Date:      September 2018
//  Copyright: Stefan Thöni
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SL.h>

//-----------------------------------------------------------------------------
class SLGLFbo
{
public:
    SLGLFbo(GLuint w,
            GLuint h,
            GLenum magFilter      = GL_NEAREST,
            GLenum minFilter      = GL_NEAREST,
            GLint  internalFormat = GL_RGB16F,
            GLint  format         = GL_FLOAT,
            GLint  wrap           = GL_REPEAT);

    ~SLGLFbo();

    void activateAsTexture(int             progId,
                           const SLstring& samplerName,
                           int             textureUnit = GL_TEXTURE0);
    SLuint width;
    SLuint height;
    SLuint frameBuffer;
    SLuint textureColorBuffer;
    SLuint attachment;
    SLuint rbo;
};
//-----------------------------------------------------------------------------