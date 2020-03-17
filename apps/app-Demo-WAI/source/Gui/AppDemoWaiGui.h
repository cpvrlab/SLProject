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
#include <GUIPreferences.h>
#include <ImGuiWrapper.h>
#include <ErlebAR.h>

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

enum class GuiAlignment
{
    TOP_LEFT,
    TOP_RIGHT,
    BOTTOM_LEFT,
    BOTTOM_RIGHT
};

class BackButton
{
public:
    BackButton(int             dotsPerInch,
               int             screenWidthPix,
               int             screenHeightPix,
               GuiAlignment    alignment,
               float           distFrameHorizMM,
               float           distFrameVertMM,
               ImVec2          buttonSizeMM,
               ButtonPressedCB pressedCB)
      : _alignment(alignment),
        _pressedCB(pressedCB)
    {
        float pixPerMM = (float)dotsPerInch / 25.4f;

        _buttonSizePix = {buttonSizeMM.x * pixPerMM, buttonSizeMM.y * pixPerMM};
        _windowSizePix = {_buttonSizePix.x + 2 * _windowPadding, _buttonSizePix.y + 2 * _windowPadding};

        //top
        if (_alignment == GuiAlignment::TOP_LEFT || _alignment == GuiAlignment::TOP_RIGHT)
        {
            _windowPos.y = distFrameVertMM * pixPerMM;
        }
        else //bottom
        {
            _windowPos.y = screenHeightPix - distFrameVertMM * pixPerMM - _windowSizePix.y;
        }

        //left
        if (_alignment == GuiAlignment::BOTTOM_LEFT || _alignment == GuiAlignment::TOP_LEFT)
        {
            _windowPos.x = distFrameHorizMM * pixPerMM;
        }
        else //right
        {
            _windowPos.x = screenWidthPix - distFrameHorizMM * pixPerMM - _windowSizePix.x;
        }
    }
    BackButton()
    {
    }

    void render()
    {
        {
            ImGuiStyle& style   = ImGui::GetStyle();
            style.WindowPadding = ImVec2(0, _windowPadding); //space l, r, b, t between window and buttons (window padding left does not work as expected)

            //back button
            ImGui::SetNextWindowPos(_windowPos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(_windowSizePix, ImGuiCond_Always);

            ImGui::Begin("", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::PushStyleColor(ImGuiCol_Button, _buttonColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _buttonColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, _buttonColorPressed);

            ImGui::NewLine();
            ImGui::SameLine(_windowPadding);
            if (ImGui::Button("back", _buttonSizePix))
            {
                if (_pressedCB)
                    _pressedCB();
            }

            ImGui::PopStyleColor(3);

            ImGui::End();
        }
    }

private:
    ImVec4 _buttonColor = {BFHColors::OrangePrimary.r,
                           BFHColors::OrangePrimary.g,
                           BFHColors::OrangePrimary.b,
                           BFHColors::OrangePrimary.a};

    ImVec4 _buttonColorPressed = {BFHColors::GrayLogo.r,
                                  BFHColors::GrayLogo.g,
                                  BFHColors::GrayLogo.b,
                                  BFHColors::GrayLogo.a};

    GuiAlignment _alignment;
    ImVec2       _windowPos;
    //calculated sized of dialogue
    ImVec2 _windowSizePix;
    //button size (in pixel)
    ImVec2 _buttonSizePix;
    //distance between button border and window border
    int _windowPadding = 2.f;

    ButtonPressedCB _pressedCB = nullptr;
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
{
public:
    AppDemoWaiGui(std::string     appName,
                  std::string     configDir,
                  int             dotsPerInch,
                  std::string     fontPath,
                  int             windowWidthPix,
                  int             windowHeightPix,
                  ButtonPressedCB backButtonCB);
    ~AppDemoWaiGui();
    //!< Checks, if a dialog with this name already exists, and adds it if not
    void addInfoDialog(std::shared_ptr<AppDemoGuiInfosDialog> dialog);
    void clearInfoDialogs();

    void build(SLScene* s, SLSceneView* sv) override;
    void loadFonts(SLfloat fontPropDots, SLfloat fontFixedDots, std::string fontPath);

    std::unique_ptr<GUIPreferences> uiPrefs;

private:
    void buildInfosDialogs(SLScene* s, SLSceneView* sv);
    void buildMenu(SLScene* s, SLSceneView* sv);

    std::string _prefsFileName;

    //! Vector containing all info dialogs, that belong to special scenes
    std::map<std::string, std::shared_ptr<AppDemoGuiInfosDialog>> _infoDialogs;

    BackButton _backButton;
};
//-----------------------------------------------------------------------------
#endif
