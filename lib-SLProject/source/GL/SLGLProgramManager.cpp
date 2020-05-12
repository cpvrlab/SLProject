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

std::map<SLShaderProg, SLGLGenericProgram*> SLGLProgramManager::_programs;

//-----------------------------------------------------------------------------
SLGLGenericProgram* SLGLProgramManager::get(SLShaderProg id)
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
void SLGLProgramManager::makeProgram(SLShaderProg id)
{
    switch (id)
    {
        case SP_colorAttribute:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "ColorAttribute.vert", "Color.frag")});
            break;
        case SP_colorUniform:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "ColorUniform.vert", "Color.frag")});
            break;
        case SP_perVrtBlinn:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "PerVrtBlinn.vert", "PerVrtBlinn.frag")});
            break;
        case SP_perVrtBlinnColorAttrib:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "PerVrtBlinnColorAttrib.vert", "PerVrtBlinn.frag")});
            break;
        case SP_perVrtBlinnTex:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "PerVrtBlinnTex.vert", "PerVrtBlinnTex.frag")});
            break;
        case SP_TextureOnly:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "TextureOnly.vert", "TextureOnly.frag")});
            break;
        case SP_perPixBlinn:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "PerPixBlinn.vert", "PerPixBlinn.frag")});
            break;
        case SP_perPixBlinnTex:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "PerPixBlinnTex.vert", "PerPixBlinnTex.frag")});
            break;
        case SP_perPixCookTorrance:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "PerPixCookTorrance.vert", "PerPixCookTorrance.frag")});
            break;
        case SP_perPixCookTorranceTex:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "PerPixCookTorranceTex.vert", "PerPixCookTorranceTex.frag")});
            break;
        case SP_bumpNormal:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "BumpNormal.vert", "BumpNormal.frag")});
            break;
        case SP_bumpNormalParallax:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "BumpNormal.vert", "BumpNormalParallax.frag")});
            break;
        case SP_fontTex:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "FontTex.vert", "FontTex.frag")});
            break;
        case SP_stereoOculus:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "StereoOculus.vert", "StereoOculus.frag")});
            break;
        case SP_stereoOculusDistortion:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "StereoOculusDistortionMesh.vert", "StereoOculusDistortionMesh.frag")});
            break;
        case SP_depth:
            _programs.insert({id, new SLGLGenericProgram(nullptr, "Depth.vert", "Depth.frag")});
            break;
        default:
            SL_EXIT_MSG("SLGLProgramManager: unknown shader id!");
    }
}
//-----------------------------------------------------------------------------
