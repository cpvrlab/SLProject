//#############################################################################
//  File:      SLGLProgramManager.cpp
//  Author:    Michael Goettlicher
//  Date:      March 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <SLGLProgramManager.h>

#include <utility>

std::map<SLStdShaderProg, SLGLGenericProgram*> SLGLProgramManager::_programs;
std::string                                    SLGLProgramManager::_shaderDir;
//-----------------------------------------------------------------------------
void SLGLProgramManager::init(std::string shaderDir)
{
    _shaderDir = std::move(shaderDir);
}
//-----------------------------------------------------------------------------
SLGLGenericProgram* SLGLProgramManager::get(SLStdShaderProg id)
{
    auto it = _programs.find(id);
    if (it == _programs.end())
    {
        makeProgram(id);
    }

    return _programs[id];
}
//-----------------------------------------------------------------------------
void SLGLProgramManager::deletePrograms()
{
    for (auto it : _programs)
        delete it.second;
    _programs.clear();
}
//-----------------------------------------------------------------------------
void SLGLProgramManager::makeProgram(SLStdShaderProg id)
{
    assert(!_shaderDir.empty() && "Error in SLGLProgramManager: Please set call SLGLProgramManager::init and transfer the location of the default shader files!");

    switch (id)
    {
        case SP_colorAttribute:
            _programs.insert({id, new SLGLGenericProgram(nullptr, _shaderDir + "ColorAttribute.vert", _shaderDir + "Color.frag")});
            break;
        case SP_colorUniform:
            _programs.insert({id, new SLGLGenericProgram(nullptr, _shaderDir + "ColorUniform.vert", _shaderDir + "Color.frag")});
            break;
        case SP_TextureOnly:
            _programs.insert({id, new SLGLGenericProgram(nullptr, _shaderDir + "TextureOnly.vert", _shaderDir + "TextureOnly.frag")});
            break;
        case SP_fontTex:
            _programs.insert({id, new SLGLGenericProgram(nullptr, _shaderDir + "FontTex.vert", _shaderDir + "FontTex.frag")});
            break;
        case SP_stereoOculus:
            _programs.insert({id, new SLGLGenericProgram(nullptr, _shaderDir + "StereoOculus.vert", _shaderDir + "StereoOculus.frag")});
            break;
        case SP_stereoOculusDistortion:
            _programs.insert({id, new SLGLGenericProgram(nullptr, _shaderDir + "StereoOculusDistortionMesh.vert", _shaderDir + "StereoOculusDistortionMesh.frag")});
            break;
        case SP_errorTex:
            _programs.insert({id, new SLGLGenericProgram(nullptr, _shaderDir + "ErrorTex.vert", _shaderDir + "ErrorTex.frag")});
            break;
        case SP_depth:
            _programs.insert({id, new SLGLGenericProgram(nullptr, _shaderDir + "Depth.vert", _shaderDir + "Depth.frag")});
            break;
        default:
            SL_EXIT_MSG("SLGLProgramManager: unknown shader id!");
    }
}
//-----------------------------------------------------------------------------
