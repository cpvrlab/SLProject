#include <AreaTrackingGui.h>
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>

using namespace ErlebAR;

AreaTrackingGui::AreaTrackingGui(sm::EventHandler&          eventHandler,
                                 ErlebAR::Resources&        resources,
                                 int                        dotsPerInch,
                                 int                        screenWidthPix,
                                 int                        screenHeightPix,
                                 std::function<void(float)> transparencyChangedCB,
                                 std::string                fontPath)
  : ImGuiWrapper(resources.fonts().atlas),
    sm::EventSender(eventHandler),
    _resources(resources),
    _transparencyChangedCB(transparencyChangedCB)
{
    resize(screenWidthPix, screenHeightPix);
    float bigTextH = _resources.style().headerBarTextH * (float)_headerBarH;
    //load fonts for big ErlebAR text and verions text
    SLstring ttf = fontPath + "Roboto-Medium.ttf";

    if (Utils::fileExists(ttf))
    {
        _fontBig = _context->IO.Fonts->AddFontFromFileTTF(ttf.c_str(), bigTextH);
    }
    else
        Utils::warnMsg("AreaTrackingGui", "font does not exist!", __LINE__, __FILE__);
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
                             _fontBig,
                             _buttonRounding,
                             buttonSize,
                             _resources.textures.texIdBackArrow,
                             _spacingBackButtonToText,
                             _area.name,
                             [&]() { sendEvent(new GoBackEvent()); });

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
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, _resources.style().whiteColor);
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, _resources.style().whiteColor);
        ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, buttonSize);
        ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, _buttonRounding);
        if (ImGui::VSliderFloat("##AreaTrackingGui_verticalSlider", ImVec2(buttonSize, _contentH * 0.7), &_sliderValue, 0.0f, 1.0f, ""))
        {
            if (_transparencyChangedCB)
                _transparencyChangedCB(_sliderValue);
        }

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(2);

        ImGui::EndChild();
        ImGui::End();

        ImGui::PopStyleColor(1);
        ImGui::PopStyleVar(7);
    }

    //ImGui::ShowMetricsWindow();

    //debug: draw log window
    _resources.logWinDraw();
}

void AreaTrackingGui::initArea(ErlebAR::Area area)
{
    _area = area;
}
