#include <AboutGui.h>
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>

using namespace ErlebAR;

AboutGui::AboutGui(const ImGuiEngine&  imGuiEngine,
                   sm::EventHandler&   eventHandler,
                   ErlebAR::Resources& resources,
                   int                 dotsPerInch,
                   int                 screenWidthPix,
                   int                 screenHeightPix)
  : ImGuiWrapper(imGuiEngine.context(), imGuiEngine.renderer()),
    sm::EventSender(eventHandler),
    _resources(resources)
{
    resize(screenWidthPix, screenHeightPix);
}

AboutGui::~AboutGui()
{
}

void AboutGui::onShow()
{
    _panScroll.enable();
}

void AboutGui::onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH, scr2fbX, scr2fbY);
}

void AboutGui::resize(int scrW, int scrH)
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

void AboutGui::build(SLScene* s, SLSceneView* sv)
{
    //header bar
    float buttonSize = _resources.style().headerBarButtonH * _headerBarH;

    ErlebAR::renderHeaderBar("AboutGui",
                             _screenW,
                             _headerBarH,
                             _resources.style().headerBarBackgroundColor,
                             _resources.style().headerBarTextColor,
                             _resources.style().headerBarBackButtonColor,
                             _resources.style().headerBarBackButtonPressedColor,
                             _resources.fonts().headerBar,
                             _buttonRounding,
                             buttonSize,
                             _resources.textures.texIdBackArrow,
                             _spacingBackButtonToText,
                             _resources.strings().about(),
                             [&]() { sendEvent(new GoBackEvent()); });

    //content
    {
        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);
        ImGuiWindowFlags childWindowFlags = ImGuiWindowFlags_NoTitleBar |
                                            ImGuiWindowFlags_NoMove |
                                            ImGuiWindowFlags_AlwaysAutoResize |
                                            ImGuiWindowFlags_NoBringToFrontOnFocus |
                                            ImGuiWindowFlags_NoScrollbar /*|
                                            ImGuiWindowFlags_NoScrollWithMouse*/
          ;
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

        ImGui::Begin("AboutGui_content", nullptr, windowFlags);
        ImGui::BeginChild("AboutGui_content_child", ImVec2(0, 0), false, childWindowFlags);

        //general
        ImGui::PushFont(_resources.fonts().heading);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
        ImGui::Text(_resources.strings().general());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + _textWrapW);
        ImGui::PushFont(_resources.fonts().standard);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textStandardColor);
        ImGui::Text(_resources.strings().generalContent(), _textWrapW);

        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
        static int       lines = 100;
        ImGuiListClipper clipper(lines);
        while (clipper.Step())
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++)
                ImGui::Text("%i The quick brown fox jumps over the lazy dog", i);
        ImGui::PopStyleVar();

        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGui::PopTextWrapPos();

        ImGui::Separator();

        //developers
        ImGui::PushFont(_resources.fonts().heading);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
        ImGui::Text(_resources.strings().developers());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::PushFont(_resources.fonts().standard);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textStandardColor);
        ImGui::Text(_resources.strings().developerNames(), _textWrapW);
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::Separator();

        //credits
        //..

        ImGui::EndChild();
        ImGui::End();

        ImGui::PopStyleColor(1);
        ImGui::PopStyleVar(7);
    }

    //ImGui::ShowMetricsWindow();

    //debug: draw log window
    _resources.logWinDraw();
}
