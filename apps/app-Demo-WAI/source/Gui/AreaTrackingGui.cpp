#include <AreaTrackingGui.h>
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>

using namespace ErlebAR;

AreaTrackingGui::AreaTrackingGui(const ImGuiEngine&         imGuiEngine,
                                 sm::EventHandler&          eventHandler,
                                 ErlebAR::Resources&        resources,
                                 int                        dotsPerInch,
                                 int                        screenWidthPix,
                                 int                        screenHeightPix,
                                 std::function<void(float)> transparencyChangedCB)
  : ImGuiWrapper(imGuiEngine.context(), imGuiEngine.renderer()),
    sm::EventSender(eventHandler),
    _resources(resources),
    _transparencyChangedCB(transparencyChangedCB)
{
    resize(screenWidthPix, screenHeightPix);
}

AreaTrackingGui::~AreaTrackingGui()
{
}

void AreaTrackingGui::onShow()
{
    _panScroll.enable();
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
    _windowPaddingContent    = _resources.style().windowPaddingContent * _screenH;
    _itemSpacingContent      = _resources.style().itemSpacingContent * _screenH;
}

void AreaTrackingGui::build(SLScene* s, SLSceneView* sv)
{
    //header bar
    float buttonSize = _resources.style().headerBarButtonH * _headerBarH;

    ErlebAR::renderHeaderBar("AreaTrackingGui",
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
                             [&]() { sendEvent(new GoBackEvent("AreaTrackingGui")); });

    //content
    {
        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);
        ImGuiWindowFlags childWindowFlags = ImGuiWindowFlags_NoTitleBar |
                                            ImGuiWindowFlags_NoMove |
                                            ImGuiWindowFlags_AlwaysAutoResize |
                                            ImGuiWindowFlags_NoBackground |
                                            ImGuiWindowFlags_NoBringToFrontOnFocus |
                                            ImGuiWindowFlags_NoScrollbar;
        ImGuiWindowFlags windowFlags = childWindowFlags |
                                       ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().backgroundColorPrimary);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(_windowPaddingContent, _windowPaddingContent));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(_windowPaddingContent, _windowPaddingContent));

        ImGui::Begin("AreaTrackingGui_content", nullptr, windowFlags);
        ImGui::BeginChild("AreaTrackingGui_content_child", ImVec2(0, 0), false, childWindowFlags);

        ImGui::PushStyleColor(ImGuiCol_FrameBg, _resources.style().frameBgColor);
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, _resources.style().frameBgColor);
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, _resources.style().frameBgActiveColor);
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, _resources.style().headerBarTextColor);
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, _resources.style().headerBarTextColor);
        ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, buttonSize);
        ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, _buttonRounding);
        if (ImGui::VSliderFloat("##AreaTrackingGui_verticalSlider", ImVec2(buttonSize, _contentH * 0.7f), &_sliderValue, 0.0f, 1.0f, ""))
        {
            if (_transparencyChangedCB)
                _transparencyChangedCB(_sliderValue);
        }

        if (_isLoading)
        {
            const float spinnerRadius = _headerBarH;
            ImVec2      spinnerPos    = {(0.5f * _screenW) - spinnerRadius, (0.5f * _screenH) - spinnerRadius};
            //ImGui::SetCursorPos(spinnerPos);
            ErlebAR::waitingSpinner("spinnerLocationMapGui",
                                    spinnerPos,
                                    spinnerRadius,
                                    _resources.style().waitingSpinnerMainColor,
                                    _resources.style().waitingSpinnerBackDropColor,
                                    13,
                                    10.f);
        }

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(2);

        ImGui::EndChild();
        ImGui::End();

        ImGui::PopStyleColor(1);
        ImGui::PopStyleVar(7);

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
    }

    //ImGui::ShowMetricsWindow();

    //debug: draw log window
    _resources.logWinDraw();
}

void AreaTrackingGui::initArea(ErlebAR::Area area)
{
    _area = area;
}
