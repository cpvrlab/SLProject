#include <AreaTrackingGui.h>
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>
#include <Utils.h>
#include <sens/SENSSimHelper.h>

using namespace ErlebAR;

AreaTrackingGui::AreaTrackingGui(const ImGuiEngine&                  imGuiEngine,
                                 sm::EventHandler&                   eventHandler,
                                 ErlebAR::Config&                    config,
                                 int                                 dotsPerInch,
                                 int                                 screenWidthPix,
                                 int                                 screenHeightPix,
                                 std::function<void(float)>          transparencyChangedCB,
                                 std::string                         erlebARDir,
                                 std::function<SENSSimHelper*(void)> getSimHelperCB)
  : ImGuiWrapper(imGuiEngine.context(), imGuiEngine.renderer()),
    sm::EventSender(eventHandler),
    _config(config),
    _resources(config.resources()),
    _transparencyChangedCB(transparencyChangedCB),
    _erlebARDir(erlebARDir),
    _getSimHelper(getSimHelperCB)
{
    resize(screenWidthPix, screenHeightPix);
}

AreaTrackingGui::~AreaTrackingGui()
{
}

void AreaTrackingGui::onShow()
{
    _panScroll.enable();
    _opacityController.resetVisible();
    if (_simHelperGui)
        _simHelperGui->reset();
    _infoText.clear();
}

void AreaTrackingGui::onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH, scr2fbX, scr2fbY);
}

void AreaTrackingGui::resize(int scrW, int scrH)
{
    _screenW = (float)scrW;
    _screenH = (float)scrH;

    _headerBarH              = _resources.style().headerBarPercH * _screenH;
    _contentH                = _screenH - _headerBarH;
    _contentStartY           = _headerBarH;
    _spacingBackButtonToText = _resources.style().headerBarSpacingBB2Text * _headerBarH;
    _buttonRounding          = _resources.style().buttonRounding * _screenH;
    _textWrapW               = 0.9f * _screenW;
}

void AreaTrackingGui::mouseDown(SLMouseButton button, bool doNotDispatch)
{
    _opacityController.mouseDown(doNotDispatch);
}

void AreaTrackingGui::mouseMove(bool doNotDispatch)
{
    //In this case we reset if event was already dispatched by imgui, e.g.
    //when the user moves the slider, we dont want the ui to hide
    _opacityController.mouseMove();
}

void AreaTrackingGui::mouseUp(SLMouseButton button, bool doNotDispatch)
{
    if (button == MB_left)
        _opacityController.mouseUp(doNotDispatch);
}

void AreaTrackingGui::build(SLScene* s, SLSceneView* sv)
{
    _opacityController.update();
    float         opacity            = _opacityController.opacity();
    const ImVec4& cHB                = _resources.style().headerBarTextColor;
    ImVec4        sliderGrabCol      = {cHB.x, cHB.y, cHB.z, cHB.w * opacity};
    const ImVec4& cFB                = _resources.style().frameBgColor;
    ImVec4        frameBgColor       = {cFB.x, cFB.y, cFB.z, cFB.w * opacity};
    const ImVec4& cFBA               = _resources.style().frameBgActiveColor;
    ImVec4        frameBgActiveColor = {cFBA.x, cFBA.y, cFBA.z, cFBA.w * opacity};

    //header bar
    float buttonSize = _resources.style().headerBarButtonH * _headerBarH;

    ErlebAR::renderHeaderBar(
      "AreaTrackingGui",
      _screenW,
      _headerBarH,
      _resources.style().headerBarBackgroundTranspColor,
      _resources.style().headerBarTextColor,
      _resources.style().headerBarBackButtonTranspColor,
      _resources.style().headerBarBackButtonPressedTranspColor,
      _resources.fonts().headerBar,
      _buttonRounding,
      buttonSize,
      _resources.textures.texIdBackArrow,
      _spacingBackButtonToText,
      _area.name.c_str(),
      [&]() { sendEvent(new GoBackEvent("AreaTrackingGui")); },
      opacity);

    //content
    {
        ImGuiWindowFlags childWindowFlags = ImGuiWindowFlags_NoTitleBar |
                                            ImGuiWindowFlags_NoMove |
                                            ImGuiWindowFlags_AlwaysAutoResize |
                                            ImGuiWindowFlags_NoBackground |
                                            ImGuiWindowFlags_NoBringToFrontOnFocus |
                                            ImGuiWindowFlags_NoScrollbar;
        ImGuiWindowFlags windowFlags = childWindowFlags |
                                       ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, buttonSize);
        ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, _buttonRounding);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().backgroundColorPrimary);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, frameBgColor);
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, frameBgColor);
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, frameBgActiveColor);
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, sliderGrabCol);
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, sliderGrabCol);

        //slider
        {
            ImVec2 sliderSize(buttonSize, _contentH * 0.7f);
            float  sliderPos = _resources.style().windowPaddingContent * _screenH;
            ImGui::SetNextWindowPos(ImVec2(sliderPos, _contentStartY + sliderPos), ImGuiCond_Always);
            ImGui::SetNextWindowSize(sliderSize, ImGuiCond_Always);

            ImGui::Begin("AreaTrackingGui_slider", nullptr, windowFlags);
            ImGui::BeginChild("AreaTrackingGui_slider_child", ImVec2(0, 0), false, childWindowFlags);

            if (ImGui::VSliderFloat("##AreaTrackingGui_verticalSlider", sliderSize, &_sliderValue, 0.0f, 1.0f, ""))
            {
                if (_transparencyChangedCB)
                    _transparencyChangedCB(_sliderValue);
            }

            ImGui::EndChild();
            ImGui::End();
        }

        //loading indicator
        if (_isLoading)
        {
            const float spinnerRadius = _headerBarH;
            ImVec2      spinnerPos    = {(0.5f * _screenW) - spinnerRadius, (0.5f * _screenH) - spinnerRadius};

            ImVec2 sliderSize(buttonSize, _contentH * 0.7f);
            float  sliderPos = _resources.style().windowPaddingContent * _screenH;
            ImGui::SetNextWindowPos(spinnerPos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(spinnerRadius * 2.f, spinnerRadius * 2.f), ImGuiCond_Always);

            ImGui::Begin("AreaTrackingGui_spinner", nullptr, windowFlags);
            ImGui::BeginChild("AreaTrackingGui_spinner_child", ImVec2(0, 0), false, childWindowFlags);

            ErlebAR::waitingSpinner("spinnerLocationMapGui",
                                    spinnerPos,
                                    spinnerRadius,
                                    _resources.style().waitingSpinnerMainColor,
                                    _resources.style().waitingSpinnerBackDropColor,
                                    13,
                                    10.f);

            ImGui::EndChild();
            ImGui::End();
        }

        if (!_infoText.empty() && _config.developerMode)
        {
            ImGuiWindowFlags infoBarWinFlags = ImGuiWindowFlags_NoTitleBar |
                                               ImGuiWindowFlags_NoMove |
                                               ImGuiWindowFlags_AlwaysAutoResize |
                                               ImGuiWindowFlags_NoScrollbar;

            ImGui::PushFont(_resources.fonts().heading);

            float  winPadding = _resources.style().windowPaddingContent * _screenH;
            float  wrapW      = _screenW - (2.f * winPadding);
            ImVec2 textSize   = ImGui::CalcTextSize(_infoText.c_str(), nullptr, false, wrapW);
            float  infoBarH   = textSize.y + 2.f * winPadding;

            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(winPadding * 0.8f, winPadding));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().headerBarBackgroundTranspColor);

            ImGui::SetNextWindowPos(ImVec2(0, _screenH - infoBarH), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(_screenW, infoBarH), ImGuiCond_Always);

            ImGui::Begin("AreaTrackingGui_userGuidanceText", nullptr, infoBarWinFlags);
            ImGui::PushTextWrapPos(wrapW);
            ImGui::TextUnformatted(_infoText.c_str());
            ImGui::PopTextWrapPos();
            ImGui::End();

            ImGui::PopStyleVar(1);
            ImGui::PopStyleColor(1);
            ImGui::PopFont();
        }

        if (_showAlignImage && _areaAlignTexture != 0)
        {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            {
                ImGui::SetNextWindowPos(ImVec2(0.1 * _screenW, 0.1 * _screenH), ImGuiCond_Always);
                ImGui::SetNextWindowSize(ImVec2(0.8 * _screenW, 0.8 * _screenH), ImGuiCond_Always);
                ImGui::Begin("AreaTrackingGui_areaAlignTexture", nullptr, windowFlags | ImGuiWindowFlags_NoBringToFrontOnFocus);
                ImVec2 uv0(0.0f, 0.0f);
                ImVec2 uv1(1.f, 1.f);
                ImVec4 col(1.f, 1.f, 1.f, _areaAlighTextureBlending);
                ImGui::Image((void*)(intptr_t)_areaAlignTexture, ImVec2(0.8 * _screenW, 0.8 * _screenH), uv0, uv1, col);
                ImGui::End();
            }
            ImGui::PopStyleVar(1);
        }

        ImGui::PopStyleColor(6);
        ImGui::PopStyleVar(9);
    }

    //error message
    if (!_errorMsg.empty())
    {
        // Calculate window position for dynamic status bar at the bottom of the main window
        ImGuiWindowFlags window_flags = 0;
        window_flags |= ImGuiWindowFlags_NoTitleBar;
        window_flags |= ImGuiWindowFlags_NoResize;
        float w = (float)_screenW - 10;

        ImFont* font = _resources.fonts().tiny;
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
        ImGui::PushFont(font);

        ImVec2 size = ImGui::CalcTextSize(_errorMsg.c_str(), nullptr, true, w);
        float  btnH = _screenH * 0.2;
        float  h    = size.y + ImGui::GetStyle().WindowPadding.y * 2.f + ImGui::GetStyle().ItemSpacing.y + btnH;

        ImGui::SetNextWindowPos(ImVec2(5, (_screenH * 0.5f) - (h * 0.5f)));
        ImGui::SetNextWindowSize(ImVec2(w, h));

        ImGui::Begin("Error", nullptr, window_flags);

        ImGui::TextWrapped("%s", _errorMsg.c_str());

        if (ImGui::Button("Okay##AreaTrackingGui", ImVec2(w * 0.2, btnH)))
        {
            _errorMsg.clear();
        }

        ImGui::End();

        ImGui::PopStyleColor();
        ImGui::PopFont();
    }

    /*
    ImGui::PushFont(_resources.fonts().tiny);
    ImGui::ShowMetricsWindow();
    ImGui::PopFont();
     */

    //debug: draw log window
    _config.logWinDraw();

    if (_config.developerMode && _config.simulatorMode && _getSimHelper)
    {
        if (!_simHelperGui)
            _simHelperGui = std::make_unique<SimHelperGui>(_resources.fonts().tiny, _resources.fonts().standard, "AreaTrackingGui", _screenH);
        _simHelperGui->render(_getSimHelper());
    }
}

void AreaTrackingGui::showInfoText(const std::string& str)
{
    _infoText = str;
}

void AreaTrackingGui::showImageAlignTexture(float alpha)
{
    if (alpha < 0.01)
    {
        _showAlignImage = false;
        return;
    }
    _showAlignImage           = true;
    _areaAlighTextureBlending = alpha;
}

void AreaTrackingGui::initArea(const ErlebAR::Area& area)
{
    _area = area;
    int w, h;

    ErlebAR::deleteTexture(_areaAlignTexture);
    if (Utils::fileExists(_erlebARDir + area.relocAlignImage))
    {
        _areaAlignTexture = ErlebAR::loadTexture(_erlebARDir + area.relocAlignImage,
                                                 false,
                                                 true,
                                                 _screenW / _screenH,
                                                 w,
                                                 h);
    }
}
