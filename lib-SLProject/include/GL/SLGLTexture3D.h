//#############################################################################
//  File:      SLGLTexture3D.h
//  Author:    Stefan Thöni
//  Date:      September 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Stefan Thöni
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLTEXTURE3D_H
#define SLGLTEXTURE3D_H

#include <SLGLTexture.h>
#include <SLVector.h>

//-----------------------------------------------------------------------------
class SLGLTexture3D
{
public:
    SLuchar* textureBuffer = nullptr;
    SLuint   textureID;

    SLGLTexture3D(
      const SLVfloat& textureBuffer, // Is there a SL implementation for this?
      SLint           width,
      SLint           height,
      SLint           depth,
      SLbool          generateMipmaps = true);

    void activate(SLint    shaderProgram,
                  SLstring glSamplerName,
                  SLint    textureUnit = GL_TEXTURE0);
    void clear(SLVec4f clearColor);

private:
    SLint _width;
    SLint _height;
    SLint _depth;
};
//-----------------------------------------------------------------------------
#endif //SLGLTEXTURE3D_H
