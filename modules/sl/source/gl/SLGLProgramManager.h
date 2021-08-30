//#############################################################################
//  File:      SLGLProgramManager.h
//  Date:      March 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLPROGRAM_MANAGER_H
#define SLGLPROGRAM_MANAGER_H

#include <string>
#include <map>

using std::string;

class SLGLProgramGeneric;

//-----------------------------------------------------------------------------
//! Enumeration for standard shader programs
enum SLStdShaderProg
{
    SP_colorAttribute = 0,
    SP_colorUniform,
    SP_TextureOnly,
    SP_TextureOnlyExternal,
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
 * done at the first use. ONLY shader programs that are scene independent
 * should be stored here. Shader programs that depend e.g. on the number of
 * lights must be created at scene loading time and deallocation at scene
 * destruction.
 */
class SLGLProgramManager
{
public:
    //! Init by providing path to standard shader files
    static void init(string shaderPath, string configPath);

    //! Get program reference for given id
    static SLGLProgramGeneric* get(SLStdShaderProg id);

    //! Delete all instantiated programs
    static void deletePrograms();

    //! Returns the size of the program map
    static size_t size() { return _programs.size(); }

    //! Contains the global shader path
    static string shaderPath;

    //! Contains the global writable configuration path;
    static string configPath;

private:
    //! Make a program if it is not contained in _programs
    static void makeProgram(SLStdShaderProg id);

    //! Instantiated programs
    static std::map<SLStdShaderProg, SLGLProgramGeneric*> _programs;
};
//-----------------------------------------------------------------------------

#endif
