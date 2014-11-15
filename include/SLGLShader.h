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

#include <stdafx.h>
#include <SLEnums.h>

//-----------------------------------------------------------------------------
//! Encapsulation of an OpenGL shader object
/*!
The SLGLShader class represents a shader object of the OpenGL Shading Language
(GLSL). It can load & compile an GLSL shader file and is later on attached
to an OpenGL shader program (SLGLProgram).
*/
class SLGLShader : public SLObject
{  
    friend class SLGLProgram;
    public:
                            SLGLShader(SLstring filename, 
                                       SLShaderType shaderType);
                           ~SLGLShader();
                       
            SLbool          createAndCompile();
            void            load(SLstring filename);
            void            loadFromMemory(SLstring program);
            SLShaderType    shaderType() {return _type;}

         
    protected:         
            SLShaderType    _type;      //!< Shader type enumeration
            SLuint          _objectGL;  //!< Program Object
            SLstring        _code;      //!< ASCII Source-Code
            SLstring        _file;      //!< Path & filename of shader
                            
};
//-----------------------------------------------------------------------------
#endif // SLSHADEROBJECT_H
