//#############################################################################
//  File:      SLGLDepthBuffer.h
//  Purpose:   Uses an OpenGL framebuffer object as a depth-buffer
//  Date:      May 2020
//  Authors:   Michael Schertenleib
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLState.h>
#include <SLGLDepthBuffer.h>
#include <Profiler.h>

//-----------------------------------------------------------------------------
/*!
 * Constructor for OpenGL depth buffer framebuffer used in shadow mapping
 * @param dimensions 2D vector pixel dimensions
 * @param magFilter OpenGL magnification filter enum
 * @param minFilter OpenGL minification filter enum
 * @param wrap OpenGL texture wrapping enum
 * @param borderColor
 * @param target OpenGL texture target enum GL_TEXTURE_2D or GL_TEXTURE_CUBE_MAP
 * @param name Name of the depth buffer
 */
SLGLDepthBuffer::SLGLDepthBuffer(const SLVec2i& dimensions,
                                 SLenum         magFilter,
                                 SLenum         minFilter,
                                 SLint          wrap,
                                 SLfloat        borderColor[],
                                 SLenum         target,
                                 SLstring       name)
  : SLObject(name),
    _dimensions(dimensions),
    _target(target)
{
    PROFILE_FUNCTION();

    assert(target == GL_TEXTURE_2D || target == GL_TEXTURE_CUBE_MAP);
    SLGLState* stateGL = SLGLState::instance();

    // Init framebuffer.
    glGenFramebuffers(1, &_fboID);
    GET_GL_ERROR;

    bind();

    glGenTextures(1, &_texID);
    stateGL->bindTexture(target, _texID);
    GET_GL_ERROR;

    // Texture parameters.
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, magFilter);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap);

#ifndef SL_GLES
    if (borderColor != nullptr)
        glTexParameterfv(target, GL_TEXTURE_BORDER_COLOR, borderColor);
#endif

    GET_GL_ERROR;

    if (target == GL_TEXTURE_2D)
    {
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_DEPTH_COMPONENT32F,
                     _dimensions.x,
                     _dimensions.y,
                     0,
                     GL_DEPTH_COMPONENT,
                     GL_FLOAT,
                     nullptr);
        GET_GL_ERROR;

        // Attach texture to framebuffer.
        glFramebufferTexture2D(GL_FRAMEBUFFER,
                               GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D,
                               _texID,
                               0);
        GET_GL_ERROR;
    }
    else // target is GL_TEXTURE_CUBE_MAP
    {
        for (SLint i = 0; i < 6; ++i)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                         0,
                         GL_DEPTH_COMPONENT32F,
                         _dimensions.x,
                         _dimensions.y,
                         0,
                         GL_DEPTH_COMPONENT,
                         GL_FLOAT,
                         nullptr);
            GET_GL_ERROR;

            // Attach texture to framebuffer.
            glFramebufferTexture2D(GL_FRAMEBUFFER,
                                   GL_DEPTH_ATTACHMENT,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   _texID,
                                   0);
            GET_GL_ERROR;
        }
    }

#ifndef SL_GLES
    glDrawBuffer(GL_NONE);
    GET_GL_ERROR;
#endif

    glReadBuffer(GL_NONE);
    GET_GL_ERROR;

    unbind();

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        SL_LOG("FBO failed to initialize correctly.");
    GET_GL_ERROR;
}

//-----------------------------------------------------------------------------
SLGLDepthBuffer::~SLGLDepthBuffer()
{
    glDeleteTextures(1, &_texID);
    GET_GL_ERROR;

    glDeleteFramebuffers(1, &_fboID);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
//! Sets the active texture unit within the shader and binds the texture
/*!
 The uniform location loc must be requested before with glUniformLocation.
 The texture unit value must correspond to the number that is set with
 glUniform1i(loc, texUnit).
 @param texUnit Texture Unit value
 */
void SLGLDepthBuffer::bindActive(SLuint texUnit) const
{

    SLGLState* stateGL = SLGLState::instance();
    stateGL->activeTexture(GL_TEXTURE0 + texUnit);
    stateGL->bindTexture(_target, _texID);
}

//-----------------------------------------------------------------------------
SLfloat* SLGLDepthBuffer::readPixels() const
{
    SLfloat* depth = new SLfloat[_dimensions.y * _dimensions.x];
    glReadPixels(0,
                 0,
                 _dimensions.x,
                 _dimensions.y,
                 GL_DEPTH_COMPONENT,
                 GL_FLOAT,
                 depth);
    GET_GL_ERROR;
    return depth;
}

//-----------------------------------------------------------------------------
//! Binds the OpenGL frame buffer object for the depth buffer
void SLGLDepthBuffer::bind()
{
    // Keep the previous FB ID for later unbinding
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &_prevFboID);
    glBindFramebuffer(GL_FRAMEBUFFER, _fboID);
    GET_GL_ERROR;
}

//-----------------------------------------------------------------------------
//! Ends the usage of the depth buffer frame buffer
void SLGLDepthBuffer::unbind()
{
    // iOS does not allow binding to 0. That's why we keep the previous FB ID
    glBindFramebuffer(GL_FRAMEBUFFER, _prevFboID);
    GET_GL_ERROR;
}

//-----------------------------------------------------------------------------
//! Binds a specific texture face of a cube map depth buffer
void SLGLDepthBuffer::bindFace(SLenum face) const
{
    assert(_target == GL_TEXTURE_CUBE_MAP);
    assert(face >= GL_TEXTURE_CUBE_MAP_POSITIVE_X &&
           face <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z);

    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_DEPTH_ATTACHMENT,
                           face,
                           _texID,
                           0);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
