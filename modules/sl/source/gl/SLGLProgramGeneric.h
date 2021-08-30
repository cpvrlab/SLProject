//#############################################################################
//  File:      SLGLProgramGeneric.h
//  Authors:   Marcus Hudritsch
//  Purpose:   Defines a minimal shader program that just starts and stops the
//             shaders that are hold in the base class SLGLProgram.
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLPROGRAMMGENERIC_H
#define SLGLPROGRAMMGENERIC_H

#include <SLGLProgram.h>
#include <SLGLProgramManager.h>
#include <SLObject.h>

class SLMaterial;
class SLAssetManager;

//-----------------------------------------------------------------------------
//! Generic Shader Program class inherited from SLGLProgram
/*!
This class only provides the shader begin and end methods. It can be used for
simple GLSL shader programs with standard types of uniform variables.
*/
class SLGLProgramGeneric : public SLGLProgram
{
public:
    ~SLGLProgramGeneric() override = default;

    //! If s is not NULL, ownership of SLGLProgram is given to SLScene (automatic deletion)
    SLGLProgramGeneric(SLAssetManager* am,
                       const SLstring& vertShaderFile,
                       const SLstring& fragShaderFile)
      : SLGLProgram(am, vertShaderFile, fragShaderFile) { ; }

    //! If s is not NULL, ownership of SLGLProgram is given to SLScene (automatic deletion)
    SLGLProgramGeneric(SLAssetManager* am,
                       const SLstring& vertShaderFile,
                       const SLstring& fragShaderFile,
                       const SLstring& geomShaderFile)
      : SLGLProgram(am, vertShaderFile, fragShaderFile, geomShaderFile) { ; }

    void beginShader(SLCamera* cam, SLMaterial* mat, SLVLight* lights) override { beginUse(cam, mat, lights); }
    void endShader() override { endUse(); }
};
//-----------------------------------------------------------------------------
#endif
