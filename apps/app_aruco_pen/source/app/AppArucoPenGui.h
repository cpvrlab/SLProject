//#############################################################################
//  File:      AppArucoPenGui.h
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGUIDEMO_H
#define SLGUIDEMO_H

#include <SL.h>

class SLScene;
class SLSceneView;
class SLNode;
class SLGLTexture;
class SLProjectScene;
class SLTexColorLUT;

//-----------------------------------------------------------------------------
class AppArucoPenGui
{
public:
    static void build(SLProjectScene* s, SLSceneView* sv);
    static void buildMenuBar(SLProjectScene* s, SLSceneView* sv);
    static void buildMenuContext(SLProjectScene* s, SLSceneView* sv);
    static void loadConfig(SLint dotsPerInch);
    static void saveConfig();

    static SLbool hideUI;
    static SLbool showDockSpace;
    static SLbool showInfosTracking;
};
//-----------------------------------------------------------------------------
#endif
