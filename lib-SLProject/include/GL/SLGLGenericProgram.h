//#############################################################################
//  File:      SLGLProgramGeneric.h
//  Author:    Marcus Hudritsch
//  Purpose:   Defines a minimal shader program that just starts and stops the
//             shaders that are hold in the base class SLGLProgram.
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLGENERICPROGRAM_H
#define SLGLGENERICPROGRAM_H

#include "SLGLProgram.h"
#include "SLObject.h"

class SLMaterial;
class SLAssetManager;

//-----------------------------------------------------------------------------
//! Generic Shader Program class inherited from SLGLProgram
/*!
This class only provides the shader begin and end methods. It can be used for
simple GLSL shader programs with standard types of uniform variables.
*/
class SLGLGenericProgram : public SLGLProgram
{
public:
    virtual ~SLGLGenericProgram() {}
    //! If s is not NULL, ownership of SLGLProgram is given to SLScene (automatic deletion)
    SLGLGenericProgram(SLAssetManager* s,
                       const char*     vertShaderFile,
                       const char*     fragShaderFile)
      : SLGLProgram(s, vertShaderFile, fragShaderFile) { ; }
    //! If s is not NULL, ownership of SLGLProgram is given to SLScene (automatic deletion)
    SLGLGenericProgram(SLAssetManager* s,
                       const char*     vertShaderFile,
                       const char*     fragShaderFile,
                       const char*     geomShaderFile)
      : SLGLProgram(s, vertShaderFile, fragShaderFile, geomShaderFile) { ; }

    virtual void beginShader(SLMaterial* mat, const SLCol4f& globalAmbientLight) { beginUse(mat, globalAmbientLight); }
    virtual void endShader() { endUse(); }
};
//-----------------------------------------------------------------------------

class SLGLColorUniformProgram : public SLGLGenericProgram
{
public:
    static SLGLColorUniformProgram* instance()
    {
        static SLGLColorUniformProgram instance;
        return &instance;
    }

private:
    SLGLColorUniformProgram()
      : SLGLGenericProgram(nullptr, "ColorUniform.vert", "Color.frag")
    {
    }
};

#endif
