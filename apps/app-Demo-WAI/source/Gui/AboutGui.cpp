#include <AboutGui.h>
#include <imgui_internal.h>

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
    _bigTextH   = _resources.style().headerBarTextH * (float)_headerBarH;
    _smallTextH = _resources.style().textStandardH * (float)screenHeightPix;

    //load fonts for big ErlebAR text and verions text
    SLstring ttf = fontPath + "Roboto-Medium.ttf";

    if (Utils::fileExists(ttf))
    {
        _fontBig   = _context->IO.Fonts->AddFontFromFileTTF(ttf.c_str(), _bigTextH);
        _fontSmall = _context->IO.Fonts->AddFontFromFileTTF(ttf.c_str(), _smallTextH);
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
    _textWrapW               = 0.8f * _screenW;
}

void AboutGui::build(SLScene* s, SLSceneView* sv)
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

        ImGui::Begin("AboutGui_header", nullptr, windowFlags);

        if (ImGui::ArrowButton("AboutGui_backButton", ImGuiDir_Left))
        {
            sendEvent(new GoBackEvent());
        }
        ImGui::SameLine(0.f, _spacingBackButtonToText);
        ImGui::Text(_resources.strings().about());

        ImGui::End();

        ImGui::PopStyleVar(1);
        ImGui::PopStyleColor(5);
        ImGui::PopFont();
    }

    //content
    {
        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);
        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize |
                                       ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
        ImGuiWindowFlags childWindowFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBringToFrontOnFocus;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().backgroundColorPrimary);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.f, 20.f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(20.f, 20.f));
        //ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, _screenH * 0.05f);
        //ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, _screenH * 0.05f * 2);
        ImGui::PushFont(_fontSmall);

        ImGui::Begin("AboutGui_content", nullptr, windowFlags);

        //general
        ImGui::BeginChild("dfgdfg", ImVec2(ImGui::GetWindowContentRegionWidth(), 0), false, childWindowFlags);
        ImGui::Text(_resources.strings().general());

        //ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
        static int lines = 1000;
        //for (int i = 0; i < lines; i++)
        //    ImGui::Text("%i The quick brown fox jumps over the lazy dog", i);
        ImGuiListClipper clipper(lines);
        while (clipper.Step())
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++)
                ImGui::Text("%i The quick brown fox jumps over the lazy dog", i);
        //ImGui::PopStyleVar();

        //ImGui::TextUnformatted(_resources.strings().generalContent());
        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + _textWrapW);
        ImGui::Text(_resources.strings().generalContent(), _textWrapW);
        ImGui::PopTextWrapPos();
        ImGui::Separator();
        //ImGui::EndChild();
        //developers
        //ImGui::BeginChild("AboutGui_content_developers", ImVec2(0, 0), false, childWindowFlags);
        ImGui::Text(_resources.strings().developers());
        ImGui::Text(_resources.strings().developerNames(), _textWrapW);
        ImGui::Separator();
        ImGui::EndChild();
        //credits
        //..

        ImGui::End();

        ImGui::PopStyleColor(1);
        ImGui::PopStyleVar(2);
        ImGui::PopFont();
    }

    popStyle();
    //ImGui::ShowMetricsWindow();
}

void AboutGui::pushStyle()
{
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    //ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 20));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
}

void AboutGui::popStyle()
{
    ImGui::PopStyleVar(6);
}
