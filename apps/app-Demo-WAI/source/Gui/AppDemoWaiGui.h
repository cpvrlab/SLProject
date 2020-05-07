//#############################################################################
//  File:      AppDemoWaiGui.h
//  Author:    Marcus Hudritsch
//  Date:      Summer 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef APPWAIDEMOGUI_H
#define APPWAIDEMOGUI_H

#include <SL.h>
#include <string>
#include <map>
#include <memory>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiSlamLoad.h>

#include <GUIPreferences.h>
#include <ImGuiWrapper.h>
#include <ErlebAR.h>
#include <AppDemoGuiError.h>
#include <sm/EventSender.h>
#include <GuiUtils.h>
#include <Resources.h>

class SLScene;
class SLSceneView;
class SLNode;
class SLGLTexture;
class AppDemoGuiInfosDialog;
class SENSCamera;
class CVCalibration;
class SENSVideoStream;

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
class AppDemoWaiGui : public ImGuiWrapper
  , public sm::EventSender
{
public:
    AppDemoWaiGui(const ImGuiEngine&                    imGuiEngine,
                  sm::EventHandler&                     eventHandler,
                  ErlebAR::Resources&                   resources,
                  std::string                           appName,
                  int                                   dotsPerInch,
                  int                                   windowWidthPix,
                  int                                   windowHeightPix,
                  std::string                           configDir,
                  std::string                           fontPath,
                  std::string                           vocabularyDir,
                  const std::vector<std::string>&       extractorIdToNames,
                  std ::queue<WAIEvent*>&               eventQueue,
                  std::function<WAISlam*(void)>         modeGetterCB,
                  std::function<SENSCamera*(void)>      getCameraCB,
                  std::function<CVCalibration*(void)>   getCalibrationCB,
                  std::function<SENSVideoStream*(void)> getVideoFileStreamCB);
    ~AppDemoWaiGui();
    //!< Checks, if a dialog with this name already exists, and adds it if not
    void addInfoDialog(std::shared_ptr<AppDemoGuiInfosDialog> dialog);
    void clearInfoDialogs();

    void build(SLScene* s, SLSceneView* sv) override;

    std::unique_ptr<GUIPreferences> uiPrefs;

    void clearErrorMsg();
    void showErrorMsg(std::string msg);

    void setSlamParams(const SlamParams& params)
    {
        if (_guiSlamLoad)
            _guiSlamLoad->setSlamParams(params);
    }

    void onShow(); //call when gui becomes visible

private:
    //void loadFonts(SLfloat fontPropDots, SLfloat fontFixedDots, std::string fontPath);

    void buildInfosDialogs(SLScene* s, SLSceneView* sv);
    void buildMenu(SLScene* s, SLSceneView* sv);

    std::string _prefsFileName;

    //! Vector containing all info dialogs, that belong to special scenes
    std::map<std::string, std::shared_ptr<AppDemoGuiInfosDialog>> _infoDialogs;

    //BackButton _backButton;

    std::shared_ptr<AppDemoGuiError>    _errorDial;
    std::shared_ptr<AppDemoGuiSlamLoad> _guiSlamLoad;

    //ImFont* _fontPropDots  = nullptr;
    //ImFont* _fontFixedDots = nullptr;
    ErlebAR::Resources& _resources;
};
//-----------------------------------------------------------------------------
#endif
