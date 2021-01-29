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
    ~SLAssetManager() { clear(); }

    void clear()
    {
        // delete materials
        for (auto m : _materials)
            delete m;
        _materials.clear();

        // delete textures
        for (auto t : _textures)
            delete t;
        _textures.clear();

        // delete meshes
        for (auto m : _meshes)
            delete m;
        _meshes.clear();

        // delete shader programs
        for (auto p : _programs)
            delete p;
        _programs.clear();
    }

    //! Removes the specified mesh from the meshes resource vector.
    bool removeMesh(SLMesh* mesh)
    {
        assert(mesh);
        for (SLulong i = 0; i < _meshes.size(); ++i)
        {
            if (_meshes[i] == mesh)
            {
                _meshes.erase(_meshes.begin() + i);
                return true;
            }
        }
        return false;
    }

    //! Returns the pointer to shader program if found by name
    SLGLProgram* getProgramByName(const string& programName)
    {
        for (auto sp : _programs)
            if (sp->name() == programName)
                return sp;
        return nullptr;
    }
    
    //! merge other asset manager into this
    void merge(SLAssetManager& other)
    {
        //update the assetmanager pointer for automatic program assignment
        for(SLMaterial* m : other.materials())
            m->assetManager(this);
        //transfer assets from other to this
        _meshes.insert(_meshes.end(), other.meshes().begin(), other.meshes().end());
        _materials.insert(_materials.end(), other.materials().begin(), other.materials().end());
        _textures.insert(_textures.end(), other.textures().begin(), other.textures().end());
        _programs.insert(_programs.end(), other.programs().begin(), other.programs().end());
        //clear ownership of other
        other.meshes().clear();
        other.materials().clear();
        other.textures().clear();
        other.programs().clear();
    }

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
