//#############################################################################
//  File:      SLGLProgramManager.cpp
//  Date:      March 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLProgramManager.h>
#include <SLGLProgramGeneric.h>

#include <utility>

std::map<SLStdShaderProg, SLGLProgramGeneric*> SLGLProgramManager::_programs;

string SLGLProgramManager::shaderPath;
string SLGLProgramManager::configPath;
//-----------------------------------------------------------------------------
/*!
 * @param shdrPath Path to the shader files
 * @param confPath Path to the writable config directory
 */
void SLGLProgramManager::init(string shdrPath, string confPath)
{
    shaderPath = std::move(shdrPath);
    configPath = std::move(confPath);
}
//-----------------------------------------------------------------------------
SLGLProgramGeneric* SLGLProgramManager::get(SLStdShaderProg id)
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
    assert(!shaderPath.empty() && "Error in SLGLProgramManager: Please set call SLGLProgramManager::init and transfer the location of the default shader files!");

    switch (id)
    {
        case SP_colorAttribute:
            _programs.insert({id, new SLGLProgramGeneric(nullptr, shaderPath + "ColorAttribute.vert", shaderPath + "Color.frag")});
            break;
        case SP_colorUniform:
            _programs.insert({id, new SLGLProgramGeneric(nullptr, shaderPath + "ColorUniform.vert", shaderPath + "Color.frag")});
            break;
        case SP_TextureOnly:
            _programs.insert({id, new SLGLProgramGeneric(nullptr, shaderPath + "TextureOnly.vert", shaderPath + "TextureOnly.frag")});
            break;
        case SP_TextureOnlyExternal:
            _programs.insert({id, new SLGLProgramGeneric(nullptr, shaderPath + "TextureOnlyExternal.vert", shaderPath + "TextureOnlyExternal.frag")});
            break;
        case SP_fontTex:
            _programs.insert({id, new SLGLProgramGeneric(nullptr, shaderPath + "FontTex.vert", shaderPath + "FontTex.frag")});
            break;
        case SP_errorTex:
            _programs.insert({id, new SLGLProgramGeneric(nullptr, shaderPath + "ErrorTex.vert", shaderPath + "ErrorTex.frag")});
            break;
        case SP_depth:
            _programs.insert({id, new SLGLProgramGeneric(nullptr, shaderPath + "Depth.vert", shaderPath + "Depth.frag")});
            break;
        default:
            SL_EXIT_MSG("SLGLProgramManager: unknown shader id!");
    }
}
//-----------------------------------------------------------------------------
