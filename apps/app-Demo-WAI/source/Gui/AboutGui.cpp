#include <AboutGui.h>
#include <imgui_internal.h>
#include <GuiUtils.h>

using namespace ErlebAR;

AboutGui::AboutGui(sm::EventHandler&   eventHandler,
                   ErlebAR::Resources& resources,
                   int                 dotsPerInch,
                   int                 screenWidthPix,
                   int                 screenHeightPix,
                   std::string         fontPath)
  : sm::EventSender(eventHandler),
    _resources(resources)
{
    resize(screenWidthPix, screenHeightPix);
    float bigTextH      = _resources.style().headerBarTextH * (float)_headerBarH;
    float headingTextH  = _resources.style().textHeadingH * (float)screenHeightPix;
    float standardTextH = _resources.style().textStandardH * (float)screenHeightPix;
    //load fonts for big ErlebAR text and verions text
    SLstring ttf = fontPath + "Roboto-Medium.ttf";

    if (Utils::fileExists(ttf))
    {
        _fontBig      = _context->IO.Fonts->AddFontFromFileTTF(ttf.c_str(), bigTextH);
        _fontSmall    = _context->IO.Fonts->AddFontFromFileTTF(ttf.c_str(), headingTextH);
        _fontStandard = _context->IO.Fonts->AddFontFromFileTTF(ttf.c_str(), standardTextH);
    }
    else
        Utils::warnMsg("WelcomeGui", "font does not exist!", __LINE__, __FILE__);
}

AboutGui::~AboutGui()
{
}

void AboutGui::onShow()
{
    _panScroll.enable();
}

void AboutGui::onResize(SLint scrW, SLint scrH)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH);
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
                             _fontBig,
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
        ImGui::PushFont(_fontSmall);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
        ImGui::Text(_resources.strings().general());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + _textWrapW);
        ImGui::PushFont(_fontStandard);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textStandardColor);
        ImGui::Text(_resources.strings().generalContent(), _textWrapW);

        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
        static int       lines = 1000;
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
        ImGui::PushFont(_fontSmall);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
        ImGui::Text(_resources.strings().developers());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::PushFont(_fontStandard);
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
}
