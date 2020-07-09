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
#include <SLGLGenericProgram.h>

#include <utility>

std::map<SLStdShaderProg, SLGLGenericProgram*> SLGLProgramManager::_programs;
std::string                                    SLGLProgramManager::shaderDir;
//-----------------------------------------------------------------------------
void SLGLProgramManager::init(std::string shaderPath)
{
    shaderDir = shaderPath;
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
    assert(!shaderDir.empty() && "Error in SLGLProgramManager: Please set call SLGLProgramManager::init and transfer the location of the default shader files!");

    switch (id)
    {
        case SP_colorAttribute:
            _programs.insert({id, new SLGLGenericProgram(nullptr, shaderDir + "ColorAttribute.vert", shaderDir + "Color.frag")});
            break;
        case SP_colorUniform:
            _programs.insert({id, new SLGLGenericProgram(nullptr, shaderDir + "ColorUniform.vert", shaderDir + "Color.frag")});
            break;
        case SP_TextureOnly:
            _programs.insert({id, new SLGLGenericProgram(nullptr, shaderDir + "TextureOnly.vert", shaderDir + "TextureOnly.frag")});
            break;
        case SP_fontTex:
            _programs.insert({id, new SLGLGenericProgram(nullptr, shaderDir + "FontTex.vert", shaderDir + "FontTex.frag")});
            break;
        case SP_stereoOculus:
            _programs.insert({id, new SLGLGenericProgram(nullptr, shaderDir + "StereoOculus.vert", shaderDir + "StereoOculus.frag")});
            break;
        case SP_stereoOculusDistortion:
            _programs.insert({id, new SLGLGenericProgram(nullptr, shaderDir + "StereoOculusDistortionMesh.vert", shaderDir + "StereoOculusDistortionMesh.frag")});
            break;
        case SP_errorTex:
            _programs.insert({id, new SLGLGenericProgram(nullptr, shaderDir + "ErrorTex.vert", shaderDir + "ErrorTex.frag")});
            break;
        case SP_depth:
            _programs.insert({id, new SLGLGenericProgram(nullptr, shaderDir + "Depth.vert", shaderDir + "Depth.frag")});
            break;
        default:
            SL_EXIT_MSG("SLGLProgramManager: unknown shader id!");
    }
}
//-----------------------------------------------------------------------------
