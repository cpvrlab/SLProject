//#############################################################################
//  File:      SLGLShader.h
//  Author:    Marcus Hudritsch
//             Mainly based on Martin Christens GLSL Tutorial
//             See http://www.clockworkcoders.com
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
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
 to an OpenGL program (SLGLProgram). Instances of SLShader are owned and
 deleted by their program (SLGLProgram).
*/
class SLGLShader : public SLObject
{
    friend class SLGLProgram;

public:
    SLGLShader();
    SLGLShader(const SLstring&     filename,
               SLShaderType type);
    ~SLGLShader() override;

    void     load(const SLstring& filename);
    void     loadFromMemory(const SLstring& program);
    SLbool   createAndCompile(SLVLight* lights);
    SLbool   createAndCompileSimple();
    SLstring removeComments(SLstring src);
    SLstring typeName();

    // Getters
    SLShaderType type() { return _type; }
    SLuint       shaderID() const { return _shaderID; }
    SLstring     code() { return _code; }

protected:
    SLShaderType _type;     //!< Shader type enumeration
    SLuint       _shaderID; //!< Program Object
    SLstring     _code;     //!< ASCII Source-Code
    SLstring     _file;     //!< Path & filename of shader
};
//-----------------------------------------------------------------------------
#endif // SLSHADEROBJECT_H
