//#############################################################################
//  File:      SLProjectScene.cpp
//  Purpose:   Declaration of the main Scene Library C-Interface.
//  Author:    Michael Goettlicher
//  Date:      March 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLProjectScene.h>
#include <SLTexFont.h>
#include <SLAssimpImporter.h>
#include <SLLightDirect.h>
#include <SLSceneView.h>
#include <SLGLProgramManager.h>
#include <AppDemo.h>

//-----------------------------------------------------------------------------
// Initialize static font pointers
SLTexFont* SLProjectScene::font07 = nullptr;
SLTexFont* SLProjectScene::font08 = nullptr;
SLTexFont* SLProjectScene::font09 = nullptr;
SLTexFont* SLProjectScene::font10 = nullptr;
SLTexFont* SLProjectScene::font12 = nullptr;
SLTexFont* SLProjectScene::font14 = nullptr;
SLTexFont* SLProjectScene::font16 = nullptr;
SLTexFont* SLProjectScene::font18 = nullptr;
SLTexFont* SLProjectScene::font20 = nullptr;
SLTexFont* SLProjectScene::font22 = nullptr;
SLTexFont* SLProjectScene::font24 = nullptr;
//-----------------------------------------------------------------------------
SLProjectScene::SLProjectScene(SLstring name, cbOnSceneLoad onSceneLoadCallback)
  : SLScene(name, onSceneLoadCallback)
{
    // font and video texture are not added to the _textures vector
    SLProjectScene::generateFonts(*SLGLProgramManager::get(SP_fontTex));
}
//----------------------------------------------------------------------------
SLProjectScene::~SLProjectScene()
{
    SLAssetManager::clear();
    SLScene::unInit();
    SLProjectScene::deleteFonts();

    SL_LOG("Destructor      : ~SLProjectScene");
}
//-----------------------------------------------------------------------------
/*! Removes the specified texture from the textures resource vector.
*/
bool SLProjectScene::deleteTexture(SLGLTexture* texture)
{
    assert(texture);
    for (SLulong i = 0; i < _textures.size(); ++i)
    {
        if (_textures[i] == texture)
        {
            delete _textures[i];
            _textures.erase(_textures.begin() + i);
            return true;
        }
    }
    return false;
}
//----------------------------------------------------------------------------
void SLProjectScene::unInit()
{
    SLScene::unInit();
    SLAssetManager::clear();
}
//-----------------------------------------------------------------------------
void SLProjectScene::onLoadAsset(const SLstring& assetFile,
                                 SLuint          processFlags)
{
    assert(false && "I commented the following lines! What influence does this have? I could not test this!");
    //AppDemo::sceneID = SID_FromFile;

    // Set scene name for new scenes
    if (!_root3D)
        name(Utils::getFileName(assetFile));

    // Try to load asset and add it to the scene root node
    SLAssimpImporter importer;

    /////////////////////////////////////////////
    SLNode* loaded = importer.load(_animManager,
                                   this,
                                   assetFile,
                                   AppDemo::texturePath,
                                   false,
                                   true,
                                   nullptr,
                                   0.0f,
                                   nullptr,
                                   processFlags);
    /////////////////////////////////////////////

    // Add root node on empty scene
    if (!_root3D)
    {
        SLNode* scene = new SLNode("Scene");
        _root3D       = scene;
    }

    // Add loaded scene
    if (loaded)
        _root3D->addChild(loaded);

    // Add directional light if no light was in loaded asset
    if (_lights.empty())
    {
        SLAABBox       boundingBox = _root3D->updateAABBRec();
        SLfloat        arrowLength = boundingBox.radiusWS() > FLT_EPSILON
                                       ? boundingBox.radiusWS() * 0.1f
                                       : 0.5f;
        SLLightDirect* light       = new SLLightDirect(this, this, 0, 0, 0, arrowLength, 1.0f, 1.0f, 1.0f);
        SLVec3f        pos         = boundingBox.maxWS().isZero()
                                       ? SLVec3f(1, 1, 1)
                                       : boundingBox.maxWS() * 1.1f;
        light->translation(pos);
        light->lookAt(pos - SLVec3f(1, 1, 1));
        light->attenuation(1, 0, 0);
        _root3D->addChild(light);
        _root3D->aabb()->reset(); // rest aabb so that it is recalculated
    }

    // call onInitialize on all scene views
    //for (auto sv : _sceneViews)
    //{
    //    if (sv != nullptr)
    //    {
    //        sv->onInitialize();
    //    }
    //}
}
//-----------------------------------------------------------------------------
//! Generates all static fonts
void SLProjectScene::generateFonts(SLGLProgram& fontTexProgram)
{
    font07 = new SLTexFont(AppDemo::fontPath + "Font07.png", &fontTexProgram);
    assert(font07);
    font08 = new SLTexFont(AppDemo::fontPath + "Font08.png", &fontTexProgram);
    assert(font08);
    font09 = new SLTexFont(AppDemo::fontPath + "Font09.png", &fontTexProgram);
    assert(font09);
    font10 = new SLTexFont(AppDemo::fontPath + "Font10.png", &fontTexProgram);
    assert(font10);
    font12 = new SLTexFont(AppDemo::fontPath + "Font12.png", &fontTexProgram);
    assert(font12);
    font14 = new SLTexFont(AppDemo::fontPath + "Font14.png", &fontTexProgram);
    assert(font14);
    font16 = new SLTexFont(AppDemo::fontPath + "Font16.png", &fontTexProgram);
    assert(font16);
    font18 = new SLTexFont(AppDemo::fontPath + "Font18.png", &fontTexProgram);
    assert(font18);
    font20 = new SLTexFont(AppDemo::fontPath + "Font20.png", &fontTexProgram);
    assert(font20);
    font22 = new SLTexFont(AppDemo::fontPath + "Font22.png", &fontTexProgram);
    assert(font22);
    font24 = new SLTexFont(AppDemo::fontPath + "Font24.png", &fontTexProgram);
    assert(font24);
}
//-----------------------------------------------------------------------------
//! Deletes all static fonts
void SLProjectScene::deleteFonts()
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
SLTexFont* SLProjectScene::getFont(SLfloat heightMM, SLint dpi)
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
