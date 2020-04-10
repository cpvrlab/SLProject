#include <SettingsGui.h>
#include <ErlebAR.h>
#include <imgui_internal.h>

SettingsGui::SettingsGui(sm::EventHandler&   eventHandler,
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

SettingsGui::~SettingsGui()
{
}

void SettingsGui::onResize(SLint scrW, SLint scrH)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH);
}

void SettingsGui::resize(int scrW, int scrH)
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

void SettingsGui::build(SLScene* s, SLSceneView* sv)
{
    pushStyle();

    //header bar with backbutton
    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _headerBarH), ImGuiCond_Always);
        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().headerBarBackgroundColor);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().headerBarTextColor);
        ImGui::PushStyleColor(ImGuiCol_Button, _resources.style().headerBarBackButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _resources.style().headerBarBackButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, _resources.style().headerBarBackButtonPressedColor);
        ImGui::PushFont(_fontBig);
        //hack for ArrowButton alignment (has to be called after font has been pushed
        float h       = _context->FontSize + _context->Style.FramePadding.y * 2.0f; //same as ImGui::GetFrameHeight()
        float spacing = 0.5f * (_headerBarH - h);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(spacing, spacing));

        ImGui::Begin("Settings_header", nullptr, windowFlags);

        if (ImGui::ArrowButton("Settings_backButton", ImGuiDir_Left))
        {
            sendEvent(new GoBackEvent());
        }
        ImGui::SameLine(0.f, _spacingBackButtonToText);
        ImGui::Text(_resources.strings().settings());

        ImGui::End();

        ImGui::PopStyleColor(5);
        ImGui::PopFont();
        ImGui::PopStyleVar(1);
    }

    //content
    {
        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);
        ImGuiWindowFlags childWindowFlags = ImGuiWindowFlags_NoTitleBar |
                                            ImGuiWindowFlags_NoMove |
                                            ImGuiWindowFlags_AlwaysAutoResize |
                                            ImGuiWindowFlags_NoBringToFrontOnFocus |
                                            ImGuiWindowFlags_NoScrollbar;
        ImGuiWindowFlags windowFlags = childWindowFlags |
                                       ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().backgroundColorPrimary);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(_windowPaddingContent, _windowPaddingContent));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(_windowPaddingContent, _windowPaddingContent));

        ImGui::Begin("Settings_content", nullptr, windowFlags);
        ImGui::BeginChild("Settings_content_child", ImVec2(0, 0), false, childWindowFlags);

        ////general
        //ImGui::PushFont(_fontSmall);
        //ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
        //ImGui::Text(_resources.strings().general());
        //ImGui::PopStyleColor();
        //ImGui::PopFont();

        //ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + _textWrapW);
        //ImGui::PushFont(_fontStandard);
        //ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textStandardColor);
        //ImGui::Text(_resources.strings().generalContent(), _textWrapW);
        //ImGui::PopStyleColor();
        //ImGui::PopFont();
        //ImGui::PopTextWrapPos();

        //ImGui::Separator();

        ////developers
        //ImGui::PushFont(_fontSmall);
        //ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
        //ImGui::Text(_resources.strings().developers());
        //ImGui::PopStyleColor();
        //ImGui::PopFont();

        //ImGui::PushFont(_fontStandard);
        //ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textStandardColor);
        //ImGui::Text(_resources.strings().developerNames(), _textWrapW);
        //ImGui::PopStyleColor();
        //ImGui::PopFont();

        //ImGui::Separator();

        ////credits
        ////..

        ImGui::EndChild();
        ImGui::End();

        ImGui::PopStyleColor(1);
        ImGui::PopStyleVar(2);
    }

    popStyle();
}

void SettingsGui::onShow()
{
    _panScroll.enable();
}

void SettingsGui::pushStyle()
{
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    //ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 20));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
}

void SettingsGui::popStyle()
{
    ImGui::PopStyleVar(6);
}
