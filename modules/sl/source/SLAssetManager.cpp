//#############################################################################
//  File:      SLAssetManager.cpp
//  Author:    Michael Goettlicher
//  Date:      Feb 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <SLAssetManager.h>

SLAssetManager::~SLAssetManager()
{
    clear();
}

//! for all assets, clear gpu data
void SLAssetManager::clear()
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

//! for all assets, clear gpu data
void SLAssetManager::deleteDataGpu()
{
    SLGLState* stateGL = SLGLState::instance();

    // check if current
    for (auto m : _materials)
    {
        if (stateGL->currentMaterial() == m)
            stateGL->currentMaterial(nullptr);
    }

    // delete textures
    for (auto t : _textures)
        t->deleteDataGpu();

    // delete meshes
    for (auto m : _meshes)
        m->deleteDataGpu();

    // delete shader programs
    for (auto p : _programs)
        ;//p->deleteDataGpu();
}

//! Removes the specified mesh from the meshes resource vector.
bool SLAssetManager::removeMesh(SLMesh* mesh)
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
SLGLProgram* SLAssetManager::getProgramByName(const string& programName)
{
    for (auto sp : _programs)
        if (sp->name() == programName)
            return sp;
    return nullptr;
}

//! merge other asset manager into this
void SLAssetManager::merge(SLAssetManager& other)
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
