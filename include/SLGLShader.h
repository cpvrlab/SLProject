//#############################################################################
//  File:      SLGLShader.h
//  Author:    Marcus Hudritsch 
//             Mainly based on Martin Christens GLSL Tutorial
//             See http://www.clockworkcoders.com
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLSHADER_H
#define SLGLSHADER_H

#include <SLEnums.h>
#include <SLObject.h>

//-----------------------------------------------------------------------------
//! Encapsulation of an OpenGL shader object
/*!
The SLGLShader class represents a shader object of the OpenGL Shading Language
(GLSL). It can load & compile an GLSL shader file and is later on attached
to an OpenGL shader program (SLGLProgram).
All shaders are written with the initial GLSL version 110 and are therefore 
backwards compatible with the compatibility profile from OpenGL 2.1 and 
OpenGL ES 2 that runs on most mobile devices. To be upwards compatible some 
modification are done in SLGLShader::createAndCompile depending on the GLSL 
version: <br>
- GLSL version > 120:
  - In vertex shaders:
    - "attribute" replaced by "in"
    - "varying" replaced by "out"
  - In fragment shaders:
    - "varying" replaced by "in"
- GLSL version 130:
  - In fragment shaders:
    - "gl_FragColor" replaced by a custom out variable
- GLSL version 140:
  - In fragment shaders:
    - "texture2D" replaced by "texture"
    - "texture3D" replaced by "texture"
    - "textureCube" replaced by "texture"
\n\n
In the OpenGL debug mode (define _GLDEBUG in SL.h) the adapted shader files 
get written out as *.debug files beside the original shader files.
*/
class SLGLShader : public SLObject
{  
    friend class SLGLProgram;
    public:
                            SLGLShader      ();
                            SLGLShader      (SLstring filename, 
                                             SLShaderType type);
                           ~SLGLShader      ();
                       
            void            load            (SLstring filename);
            void            loadFromMemory  (SLstring program);
            SLbool          createAndCompile();
            SLstring        removeComments  (SLstring src);
            SLstring        typeName        ();

            // Getters
            SLShaderType    type            () {return _type;}
            SLuint          objectGL        () {return _objectGL;}
            SLstring        code            () {return _code;}

    protected:         
            SLShaderType    _type;      //!< Shader type enumeration
            SLuint          _objectGL;  //!< Program Object
            SLstring        _code;      //!< ASCII Source-Code
            SLstring        _file;      //!< Path & filename of shader                            
};
//-----------------------------------------------------------------------------
#endif // SLSHADEROBJECT_H
