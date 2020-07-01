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
enum SLStdShaderProg
{
    SP_colorAttribute = 0,
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
    SP_depth,
    SP_errorTex
};

//-----------------------------------------------------------------------------
//! Static container for standard shader programs
/*!
 * Static container for standard shader programs that are not deleted after
 * scene deallocation. The shader program allocation and compilation will be
 * done at the first use.
 * All non standard shader programs will be attached to the asset manager.
 */
class SLGLProgramManager
{
public:
    //! Init by providing path to standard shader files
    static void init(std::string shaderDir);

    //! Get program reference for given id
    static SLGLGenericProgram* get(SLStdShaderProg id);

    //! Delete all instantiated programs
    static void deletePrograms();

    //! Returns the size of the program map
    static size_t size() { return _programs.size(); }

private:
    //! Make a program if it is not contained in _programs
    static void makeProgram(SLStdShaderProg id);

    //! Instantiated programs
    static std::map<SLStdShaderProg, SLGLGenericProgram*> _programs;
    //! Directory containing all standard shaders
    static std::string _shaderDir;
};
//-----------------------------------------------------------------------------

#endif
