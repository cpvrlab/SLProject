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
#include <SLGLGenericProgram.h>

class SLSceneView;
class SLCamera;
class SLInputManager;

class SLAssetManager
{
public:
    ~SLAssetManager()
    {
        clear();
    }

    void clear()
    {
        // delete materials
        for (auto m : _materials) delete m;
        _materials.clear();

        // delete textures
        for (auto t : _textures) delete t;
        _textures.clear();

        // delete meshes
        for (auto m : _meshes)
            delete m;
        _meshes.clear();

        // delete shader programs
        for (auto p : _programs) delete p;
        _programs.clear();
    }

    //-----------------------------------------------------------------------------
    /*! Removes the specified mesh from the meshes resource vector.
*/
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

    SLVGLProgram& programs() { return _programs; }
    SLVMaterial&  materials() { return _materials; }
    SLVGLTexture& textures() { return _textures; }
    SLVMesh&      meshes() { return _meshes; }

protected:
    SLVGLProgram _programs;  //!< Vector of all shader program pointers
    SLVMaterial  _materials; //!< Vector of all materials pointers
    SLVGLTexture _textures;  //!< Vector of all texture pointers
    SLVMesh      _meshes;    //!< Vector of all meshes
};

#endif //SLASSETMANAGER_H
