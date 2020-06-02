//#############################################################################
//  File:      SLProjectScene.h
//  Purpose:   Declaration of the main Scene Library C-Interface.
//  Author:    Michael Goettlicher
//  Date:      March 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECTSCENE_H
#define SLPROJECTSCENE_H

#include <SLScene.h>
#include <SLAssetManager.h>

class SLTexFont;

//-----------------------------------------------------------------------------
//! Class for the toplevel combination of SLScene and SLAssetManager
/*! SLProjectScene combines the scene with the asset manager. In addition it
holds the font textures that are used by the imgui-UI.
*/
class SLProjectScene : public SLScene
  , public SLAssetManager
{
public:
    SLProjectScene(SLstring name, cbOnSceneLoad onSceneLoadCallback);
    ~SLProjectScene() override;

    void unInit() override;
    bool deleteTexture(SLGLTexture* texture);

    virtual void onLoadAsset(const SLstring& assetFile,
                             SLuint          processFlags);

    // Static method & font pointers
    static void       generateFonts(SLGLProgram& fontTexProgram);
    static void       deleteFonts();
    static SLTexFont* getFont(SLfloat heightMM, SLint dpi);

    static SLTexFont* font07; //!< 7 pixel high fixed size font
    static SLTexFont* font08; //!< 8 pixel high fixed size font
    static SLTexFont* font09; //!< 9 pixel high fixed size font
    static SLTexFont* font10; //!< 10 pixel high fixed size font
    static SLTexFont* font12; //!< 12 pixel high fixed size font
    static SLTexFont* font14; //!< 14 pixel high fixed size font
    static SLTexFont* font16; //!< 16 pixel high fixed size font
    static SLTexFont* font18; //!< 18 pixel high fixed size font
    static SLTexFont* font20; //!< 20 pixel high fixed size font
    static SLTexFont* font22; //!< 22 pixel high fixed size font
    static SLTexFont* font24; //!< 24 pixel high fixed size font
};
//-----------------------------------------------------------------------------

#endif // SLPROJECTSCENE_H
