//#############################################################################
//  File:      AppDemoGui.h
//  Author:    Marcus Hudritsch
//  Date:      Summer 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef APPDEMO_GUI_H
#define APPDEMO_GUI_H

#include <SL.h>
#include <string>
#include <map>
#include <memory>
#include <AppDemoGuiInfosDialog.h>
#include <WAI.h>
#include <WAICalibration.h>
#include <AppDemoGuiPrefs.h>

class SLScene;
class SLSceneView;
class SLNode;
class SLGLTexture;
class AppDemoGuiInfosDialog;

//-----------------------------------------------------------------------------
enum SceneID
{
    Scene_None,
    Scene_Empty,
    Scene_Minimal,
    Scene_WAI
};
//-----------------------------------------------------------------------------
//! ImGui UI class for the UI of the demo applications
/* The UI is completely built within this class by calling build function
AppDemoGui::build. This build function is passed in the slCreateSceneView and
it is called in SLSceneView::onPaint in every frame.<br>
The entire UI is configured and built on every frame. That is why it is called
"Im" for immediate. See also the SLGLImGui class to see how it minimaly
integrated in the SLProject.<br>
*/
class AppDemoGui
{
    public:
    //!< Checks, if a dialog with this name already exists, and adds it if not
    static void addInfoDialog(AppDemoGuiInfosDialog* dialog);
    static void clearInfoDialogs();
    static void buildInfosDialogs(SLScene* s, SLSceneView* sv);

    static void saveConfig(GUIPreferences* prefs);
    static void loadConfig(GUIPreferences* prefs, SLint dotsPerInch);

    //! Vector containing all info dialogs, that belong to special scenes
    static std::map<std::string, AppDemoGuiInfosDialog*> _infoDialogs;
};
//-----------------------------------------------------------------------------
#endif
