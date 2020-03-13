//#############################################################################
//  File:      SLGLProgramManager.h
//  Author:    Michael Goettlicher
//  Date:      March 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLPROGRAM_MANAGER_H
#define SLGLPROGRAM_MANAGER_H

#include <SLGLGenericProgram.h>

//-----------------------------------------------------------------------------
//! Enumeration for standard shader programs
enum SLShaderProg
{
    SP_colorAttribute,
    SP_colorUniform,
    SP_perVrtBlinn,
    SP_perVrtBlinnColorAttrib,
    SP_perVrtBlinnTex,
    SP_TextureOnly,
    SP_perPixBlinn,
    SP_perPixBlinnTex,
    SP_perPixCookTorrance,
    SP_perPixCookTorranceTex,
    SP_bumpNormal,
    SP_bumpNormalParallax,
    SP_fontTex,
    SP_stereoOculus,
    SP_stereoOculusDistortion
};

//-----------------------------------------------------------------------------
class SLGLProgramManager
{
public:
    static SLGLGenericProgram* get(SLShaderProg id)
    {
        auto it = _programs.find(id);
        if (_programs.find(id) == _programs.end())
        {
            makeProgram(id);
        }

        return _programs[id];
    }

    static void deletePrograms()
    {
        for (auto it : _programs)
            delete it.second;
        _programs.clear();
    }

    ~SLGLProgramManager()
    {
        if (_programs.size())
            SL_WARN_MSG("SLGLProgramManager: you have to call deletePrograms() before closing the program!");
    }

private:
    static void makeProgram(SLShaderProg id)
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
            default:
                SL_EXIT_MSG("SLGLProgramManager: unknown shader id!");
        }
    }

    static std::map<SLShaderProg, SLGLGenericProgram*> _programs;
};
//-----------------------------------------------------------------------------

#endif
