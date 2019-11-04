//#############################################################################
//  File:      SLGLConetracerTex3D.cpp
//  Author:    Stefan Thoeni
//  Date:      September 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Stefan Thoeni
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLConetracerTex3D.h>
#include <SLGLState.h>

//-----------------------------------------------------------------------------
SLGLConetracerTex3D::SLGLConetracerTex3D(const SLVfloat& textureBuffer,
                                         SLint           width,
                                         SLint           height,
                                         SLint           depth,
                                         SLbool          generateMipmaps)
  : _width(width), _height(height), _depth(depth)
{

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_3D, textureID);
    GET_GL_ERROR;

    // Parameter options.
    const auto wrap = GL_CLAMP_TO_BORDER;
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, wrap);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, wrap);
    GET_GL_ERROR;

    const auto filter = GL_LINEAR_MIPMAP_LINEAR;
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, filter);
    GET_GL_ERROR;
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    GET_GL_ERROR;

    // Upload texture buffer.
    const int levels = 7;
    glTexStorage3D(GL_TEXTURE_3D,
                   levels,
                   GL_RGBA8,
                   _width,
                   _height,
                   _depth);
    GET_GL_ERROR;
    //std::cout << "width: " << _width << std::endl;
    //std::cout << "height: " << _height << std::endl;
    //std::cout << "depth: " << _depth << std::endl;
    //std::cout << "size: " << textureBuffer.size() << std::endl;

    glTexSubImage3D(GL_TEXTURE_3D,
                    0,
                    0,
                    0,
                    0,
                    _width,
                    _height,
                    _depth,
                    GL_RGBA,
                    GL_FLOAT,
                    &textureBuffer[0]);
    GET_GL_ERROR;

    if (generateMipmaps)
        glGenerateMipmap(GL_TEXTURE_3D);
    GET_GL_ERROR;

    glBindTexture(GL_TEXTURE_3D, 0);
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
void SLGLConetracerTex3D::activate(SLint    shaderProgram,
                                   SLstring glSamplerName,
                                   SLint    textureUnit)
{
    glActiveTexture(GL_TEXTURE0 + textureUnit);
    glBindTexture(GL_TEXTURE_3D, textureID);

    // add texture to shader:
    glUniform1i(glGetUniformLocation(shaderProgram,
                                     glSamplerName.c_str()),
                textureUnit);
}
//-----------------------------------------------------------------------------
void SLGLConetracerTex3D::clear(SLVec4f clearColor)
{
    // retrieve currently activated texture
    GLint previousBoundTextureID;
    glGetIntegerv(GL_TEXTURE_BINDING_3D, &previousBoundTextureID);

    // retrieve this texutre
    glBindTexture(GL_TEXTURE_3D, textureID);
    glClearTexImage(textureID, 0, GL_RGBA, GL_FLOAT, &clearColor);

    // rebind previous texture
    glBindTexture(GL_TEXTURE_3D, previousBoundTextureID);
}
//-----------------------------------------------------------------------------
