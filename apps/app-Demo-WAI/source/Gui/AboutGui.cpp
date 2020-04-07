#include <AboutGui.h>
#include <ErlebAR.h>
#include <imgui_internal.h>

AboutGui::AboutGui(sm::EventHandler& eventHandler,
                   int               dotsPerInch,
                   int               screenWidthPix,
                   int               screenHeightPix,
                   std::string       fontPath)
  : sm::EventSender(eventHandler)
{
    resize(screenWidthPix, screenHeightPix);
    _bigTextH   = ErlebAR::HeaderBarTextH * (float)_headerBarH;
    _smallTextH = ErlebAR::StandardTextH * (float)screenHeightPix;

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

void AboutGui::onResize(SLint scrW, SLint scrH)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH);
}

void AboutGui::resize(int scrW, int scrH)
{
    _screenW = (float)scrW;
    _screenH = (float)scrH;

    _headerBarH              = ErlebAR::HeaderBarPercH * _screenH;
    _contentH                = _screenH - _headerBarH;
    _contentStartY           = _headerBarH;
    _spacingBackButtonToText = ErlebAR::HeaderBarSpacingBB2Text * _headerBarH;
    _buttonRounding          = ErlebAR::ButtonRounding * _screenH;
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

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ErlebAR::HeaderBarBackgroundColor);
        ImGui::PushStyleColor(ImGuiCol_Text, ErlebAR::HeaderBarTextColor);
        ImGui::PushStyleColor(ImGuiCol_Button, ErlebAR::BackButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ErlebAR::BackButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ErlebAR::BackButtonPressedColor);
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
        ImGui::Text("About");

        ImGui::End();

        ImGui::PopStyleVar(1);
        ImGui::PopStyleColor(5);
        ImGui::PopFont();
    }

    //content
    {
        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);
        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ErlebAR::PrimaryBackgroundColor);

        ImGui::Begin("AboutGui_content", nullptr, windowFlags);

        ImGui::Text("Developers");
        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + _textWrapW);
        ImGui::Text("The lazy dog is a good dog. This paragraph is made to fit within %.0f pixels. Testing a 1 character word. The quick brown fox jumps over the lazy dog.", _textWrapW);
        ImGui::PopTextWrapPos();

        ImGui::End();

        ImGui::PopStyleColor(1);
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
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 20));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
}

void AboutGui::popStyle()
{
    ImGui::PopStyleVar(7);
}
