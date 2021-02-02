//#############################################################################
//  File:      SLAssetManager.h
//  Author:    Michael Goettlicher
//  Date:      Feb 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLASSETMANAGER_H
#define SLASSETMANAGER_H

#include <SL.h>
#include <SLMaterial.h>
#include <SLMesh.h>
#include <vector>
#include <SLGLProgramGeneric.h>

class SLCamera;
class SLInputManager;

//-----------------------------------------------------------------------------
//! Toplevel holder of the assets meshes, materials, textures and shader progs.
/*! This class is inherited by SLProjectScene that combines it with SLScene.
 All these assets can be shared among instances of SLScene, SLNode and SLMaterial.
 Shared assets are meshes (SLMesh), materials (SLMaterial), textures (SLGLTexture)
 and shader programs (SLGLProgram).
*/
class SLAssetManager
{
public:
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

    SLVMesh&      meshes() { return _meshes; }
    SLVMaterial&  materials() { return _materials; }
    SLVGLTexture& textures() { return _textures; }
    SLVGLProgram& programs() { return _programs; }

protected:
    SLVMesh      _meshes;    //!< Vector of all meshes
    SLVMaterial  _materials; //!< Vector of all materials pointers
    SLVGLTexture _textures;  //!< Vector of all texture pointers
    SLVGLProgram _programs;  //!< Vector of all shader program pointers
};
//-----------------------------------------------------------------------------
#endif //SLASSETMANAGER_H
