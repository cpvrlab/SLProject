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
class SLProjectScene : public SLScene
  , public SLAssetManager
{
public:
    SLProjectScene(SLstring name, cbOnSceneLoad onSceneLoadCallback);
    ~SLProjectScene();

    void unInit() override;
    bool deleteTexture(SLGLTexture* texture);

    virtual void onLoadAsset(const SLstring& assetFile,
                             SLuint          processFlags);

    // Static method & font pointers
    static void       generateFonts(SLGLProgram& fontTexProgram);
    static void       deleteFonts();
    static SLTexFont* getFont(SLfloat heightMM, SLint dpi);

    static SLTexFont* font07;
    static SLTexFont* font08;
    static SLTexFont* font09;
    static SLTexFont* font10;
    static SLTexFont* font12;
    static SLTexFont* font14;
    static SLTexFont* font16;
    static SLTexFont* font18;
    static SLTexFont* font20;
    static SLTexFont* font22;
    static SLTexFont* font24;
};
//-----------------------------------------------------------------------------

#endif // SLPROJECTSCENE_H
