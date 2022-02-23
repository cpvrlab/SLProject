//#############################################################################
//  File:      SLAssetManager.h
//  Authors:   Michael Goettlicher
//  Date:      Feb 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLASSETMANAGER_H
#define SLASSETMANAGER_H

#include <vector>

#include <SL.h>
#include <SLMaterial.h>
#include <SLMesh.h>
#include <SLGLProgramGeneric.h>

class SLTexFont;
class SLCamera;
class SLInputManager;

//-----------------------------------------------------------------------------
//! Toplevel holder of the assets meshes, materials, textures and shaders
/*! All these assets can be shared among instances of SLScene, SLNode and
 * SLMaterial. Shared assets are meshes (SLMesh), materials (SLMaterial),
 * textures (SLGLTexture) and shader programs (SLGLProgram).
 * One instance of SLAssetManager must be owned and destroyed by the app
 * (see e.g. AppDemo.h).
 */
class SLAssetManager
{
public:
    SLAssetManager(SLstring fontPath = "", bool generateStaticFonts = false)
    {
        if (!fontPath.empty() && generateStaticFonts)
        {
            // font and video texture are not added to the _textures vector
            SLAssetManager::generateFonts(fontPath,
                                          *SLGLProgramManager::get(SP_fontTex));
        }
    }
    ~SLAssetManager();

    void clear();

    //! for all assets, clear gpu data
    void deleteDataGpu();

    //! Removes the specified mesh from the meshes resource vector.
    bool removeMesh(SLMesh* mesh);

    //! Returns the pointer to shader program if found by name
    SLGLProgram* getProgramByName(const string& programName);

    //! merge other asset manager into this
    void merge(SLAssetManager& other);

    // Getters
    SLVMesh&      meshes() { return _meshes; }
    SLVMaterial&  materials() { return _materials; }
    SLVGLTexture& textures() { return _textures; }
    SLVGLProgram& programs() { return _programs; }

    // Static method & font pointers
    // These fonts can be used inside a 2D or 3D scene, not within ImGui
    static void       generateFonts(SLstring     fontPath,
                                    SLGLProgram& fontTexProgram);
    static void       deleteFonts();
    static SLTexFont* getFont(SLfloat heightMM, SLint dpi);

    static SLTexFont* font07; //!<  7 pixel high fixed size font
    static SLTexFont* font08; //!<  8 pixel high fixed size font
    static SLTexFont* font09; //!<  9 pixel high fixed size font
    static SLTexFont* font10; //!< 10 pixel high fixed size font
    static SLTexFont* font12; //!< 12 pixel high fixed size font
    static SLTexFont* font14; //!< 14 pixel high fixed size font
    static SLTexFont* font16; //!< 16 pixel high fixed size font
    static SLTexFont* font18; //!< 18 pixel high fixed size font
    static SLTexFont* font20; //!< 20 pixel high fixed size font
    static SLTexFont* font22; //!< 22 pixel high fixed size font
    static SLTexFont* font24; //!< 24 pixel high fixed size font

protected:
    SLVMesh      _meshes;    //!< Vector of all meshes
    SLVMaterial  _materials; //!< Vector of all materials pointers
    SLVGLTexture _textures;  //!< Vector of all texture pointers
    SLVGLProgram _programs;  //!< Vector of all shader program pointers
};
//-----------------------------------------------------------------------------
#endif // SLASSETMANAGER_H
