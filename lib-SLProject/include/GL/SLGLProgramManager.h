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
    SP_stereoOculusDistortion,
    SP_depth
};

//-----------------------------------------------------------------------------
class SLGLProgramManager
{
public:
    //! Get program reference for given id
    static SLGLGenericProgram* get(SLShaderProg id);
    //! Delete all instantiated programs
    static void deletePrograms();

private:
    //! Make a program if it is not contained in _programs
    static void makeProgram(SLShaderProg id);
    //! Instantiated programs
    static std::map<SLShaderProg, SLGLGenericProgram*> _programs;
};
//-----------------------------------------------------------------------------

#endif
