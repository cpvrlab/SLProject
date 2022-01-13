//#############################################################################
//  File:      SLAssetManager.cpp
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  Date:      Feb 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLAssetManager.h>
#include <SLTexFont.h>

//-----------------------------------------------------------------------------
// Initialize static font pointers
SLTexFont* SLAssetManager::font07 = nullptr;
SLTexFont* SLAssetManager::font08 = nullptr;
SLTexFont* SLAssetManager::font09 = nullptr;
SLTexFont* SLAssetManager::font10 = nullptr;
SLTexFont* SLAssetManager::font12 = nullptr;
SLTexFont* SLAssetManager::font14 = nullptr;
SLTexFont* SLAssetManager::font16 = nullptr;
SLTexFont* SLAssetManager::font18 = nullptr;
SLTexFont* SLAssetManager::font20 = nullptr;
SLTexFont* SLAssetManager::font22 = nullptr;
SLTexFont* SLAssetManager::font24 = nullptr;
//-----------------------------------------------------------------------------
SLAssetManager::~SLAssetManager()
{
    clear();
}
//-----------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------
//! for all assets, clear gpu data
void SLAssetManager::deleteDataGpu()
{
    /*
    SLGLState* stateGL = SLGLState::instance();

    // check if current
    for (auto m : _materials)
    {
        if (stateGL->currentMaterial() == m)
            stateGL->currentMaterial(nullptr);
    }
     */

    // delete textures
    for (auto t : _textures)
        t->deleteDataGpu();

    // delete meshes
    for (auto m : _meshes)
        m->deleteDataGpu();

    // delete shader programs
    for (auto p : _programs)
        ; // p->deleteDataGpu();
}
//-----------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------
//! Returns the pointer to shader program if found by name
SLGLProgram* SLAssetManager::getProgramByName(const string& programName)
{
    for (auto sp : _programs)
        if (sp->name() == programName)
            return sp;
    return nullptr;
}
//-----------------------------------------------------------------------------
//! merge other asset manager into this
void SLAssetManager::merge(SLAssetManager& other)
{
    // update the assetmanager pointer for automatic program assignment
    for (SLMaterial* m : other.materials())
        m->assetManager(this);
    // transfer assets from other to this
    _meshes.insert(_meshes.end(), other.meshes().begin(), other.meshes().end());
    _materials.insert(_materials.end(), other.materials().begin(), other.materials().end());
    _textures.insert(_textures.end(), other.textures().begin(), other.textures().end());
    _programs.insert(_programs.end(), other.programs().begin(), other.programs().end());
    // clear ownership of other
    other.meshes().clear();
    other.materials().clear();
    other.textures().clear();
    other.programs().clear();
}
//-----------------------------------------------------------------------------
//! Generates all static fonts
void SLAssetManager::generateFonts(SLstring fontPath, SLGLProgram& fontTexProgram)
{
    font07 = new SLTexFont(fontPath + "Font07.png", &fontTexProgram);
    assert(font07);
    font08 = new SLTexFont(fontPath + "Font08.png", &fontTexProgram);
    assert(font08);
    font09 = new SLTexFont(fontPath + "Font09.png", &fontTexProgram);
    assert(font09);
    font10 = new SLTexFont(fontPath + "Font10.png", &fontTexProgram);
    assert(font10);
    font12 = new SLTexFont(fontPath + "Font12.png", &fontTexProgram);
    assert(font12);
    font14 = new SLTexFont(fontPath + "Font14.png", &fontTexProgram);
    assert(font14);
    font16 = new SLTexFont(fontPath + "Font16.png", &fontTexProgram);
    assert(font16);
    font18 = new SLTexFont(fontPath + "Font18.png", &fontTexProgram);
    assert(font18);
    font20 = new SLTexFont(fontPath + "Font20.png", &fontTexProgram);
    assert(font20);
    font22 = new SLTexFont(fontPath + "Font22.png", &fontTexProgram);
    assert(font22);
    font24 = new SLTexFont(fontPath + "Font24.png", &fontTexProgram);
    assert(font24);
}
//-----------------------------------------------------------------------------
//! Deletes all static fonts
void SLAssetManager::deleteFonts()
{
    delete font07;
    font07 = nullptr;
    delete font08;
    font08 = nullptr;
    delete font09;
    font09 = nullptr;
    delete font10;
    font10 = nullptr;
    delete font12;
    font12 = nullptr;
    delete font14;
    font14 = nullptr;
    delete font16;
    font16 = nullptr;
    delete font18;
    font18 = nullptr;
    delete font20;
    font20 = nullptr;
    delete font22;
    font22 = nullptr;
    delete font24;
    font24 = nullptr;
}
//-----------------------------------------------------------------------------
//! returns nearest font for a given height in mm
SLTexFont* SLAssetManager::getFont(SLfloat heightMM, SLint dpi)
{
    SLfloat dpmm       = (SLfloat)dpi / 25.4f;
    SLfloat targetH_PX = dpmm * heightMM;

    if (targetH_PX < 7) return font07;
    if (targetH_PX < 8) return font08;
    if (targetH_PX < 9) return font09;
    if (targetH_PX < 10) return font10;
    if (targetH_PX < 12) return font12;
    if (targetH_PX < 14) return font14;
    if (targetH_PX < 16) return font16;
    if (targetH_PX < 18) return font18;
    if (targetH_PX < 20) return font20;
    if (targetH_PX < 24) return font22;
    return font24;
}
//-----------------------------------------------------------------------------
