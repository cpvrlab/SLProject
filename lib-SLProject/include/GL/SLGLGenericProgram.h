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

#include <SLGLProgram.h>
#include <SLObject.h>

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
    ~SLGLGenericProgram() override = default;

    //! If s is not NULL, ownership of SLGLProgram is given to SLScene (automatic deletion)
    SLGLGenericProgram(SLAssetManager* s,
                       const SLstring& vertShaderFile,
                       const SLstring& fragShaderFile)
      : SLGLProgram(s, vertShaderFile, fragShaderFile) { ; }

    //! If s is not NULL, ownership of SLGLProgram is given to SLScene (automatic deletion)
    SLGLGenericProgram(SLAssetManager* s,
                       const SLstring& vertShaderFile,
                       const SLstring& fragShaderFile,
                       const SLstring& geomShaderFile)
      : SLGLProgram(s, vertShaderFile, fragShaderFile, geomShaderFile) { ; }

    void beginShader(SLCamera* cam, SLMaterial* mat, SLVLight* lights) override { beginUse(cam, mat, lights); }
    void endShader() override { endUse(); }
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per vertex lighting without textures
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLGenericProgramDefault : public SLGLGenericProgram
{
public:
    static SLGLGenericProgramDefault* instance()
    {
        if (!_instance)
            _instance = new SLGLGenericProgramDefault();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }

private:
    SLGLGenericProgramDefault()
      : SLGLGenericProgram(nullptr,
                           SLApplication::shaderPath +  "PerVrtBlinn.vert",
                           SLApplication::shaderPath + "PerVrtBlinn.frag")
    {
        _name = "DefaultPerVertexProgram";
    };

    static SLGLGenericProgramDefault* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per vertex lighting with textures
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLGenericProgramDefaultTex : public SLGLGenericProgram
{
public:
    static SLGLGenericProgramDefaultTex* instance()
    {
        if (!_instance)
            _instance = new SLGLGenericProgramDefaultTex();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }

private:
    SLGLGenericProgramDefaultTex()
      : SLGLGenericProgram(nullptr,
                           SLApplication::shaderPath +  "PerVrtBlinnTex.vert",
                           SLApplication::shaderPath + "PerVrtBlinnTex.frag")
    {
        _name = "DefaultPerVertexProgramTex";
    };

    static SLGLGenericProgramDefaultTex* _instance;
};
//-----------------------------------------------------------------------------


#endif
