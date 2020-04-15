#include <TutorialGui.h>
#include <imgui_internal.h>
#include <CVImage.h>

using namespace ErlebAR;

TutorialGui::TutorialGui(sm::EventHandler&   eventHandler,
                         ErlebAR::Resources& resources,
                         int                 dotsPerInch,
                         int                 screenWidthPix,
                         int                 screenHeightPix,
                         std::string         fontPath,
                         std::string         texturePath)
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

    //load background texture
    std::string imagePath = texturePath + "earth2048_C.jpg";
    if (Utils::fileExists(imagePath))
    {
        // load texture image
        CVImage image(imagePath);
        image.flipY();
        //crop image to screen size
        image.crop((float)screenWidthPix / (float)screenHeightPix);

        _textureBackgroundW = image.width();
        _textureBackgroundH = image.height();

        // Create a OpenGL texture identifier
        glGenTextures(1, &_textureBackgroundId);
        glBindTexture(GL_TEXTURE_2D, _textureBackgroundId);

        // Setup filtering parameters for display
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Upload pixels into texture
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     image.format(),
                     (GLsizei)image.width(),
                     (GLsizei)image.height(),
                     0,
                     image.format(),
                     GL_UNSIGNED_BYTE,
                     (GLvoid*)image.data());
    }
    else
        Utils::warnMsg("TutorialGui", "imagePath does not exist!", __LINE__, __FILE__);
}

TutorialGui::~TutorialGui()
{
    if (_textureBackgroundId)
    {
        glDeleteTextures(1, &_textureBackgroundId);
        _textureBackgroundId = 0;
    }
}

void TutorialGui::onShow()
{
    _panScroll.enable();
}

void TutorialGui::onResize(SLint scrW, SLint scrH)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH);
}

void TutorialGui::resize(int scrW, int scrH)
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

void TutorialGui::build(SLScene* s, SLSceneView* sv)
{
    //background texture
    {
        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _screenH), ImGuiCond_Always);
        ImGui::Begin("TutorialGui_BackgroundTexture", nullptr, windowFlags | ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::Image((void*)(intptr_t)_textureBackgroundId, ImVec2(_screenW, _screenH));
        ImGui::End();

        ImGui::PopStyleVar(1);
    }

    pushStyle();

    //header bar with backbutton
    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _headerBarH), ImGuiCond_Always);
        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().headerBarBackgroundTranspColor);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().headerBarTextColor);
        ImGui::PushStyleColor(ImGuiCol_Button, _resources.style().headerBarBackButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _resources.style().headerBarBackButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, _resources.style().headerBarBackButtonPressedColor);
        ImGui::PushFont(_fontBig);
        //hack for ArrowButton alignment (has to be called after font has been pushed
        float h       = _context->FontSize + _context->Style.FramePadding.y * 2.0f; //same as ImGui::GetFrameHeight()
        float spacing = 0.5f * (_headerBarH - h);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(spacing, spacing));

        ImGui::Begin("TutorialGui_header", nullptr, windowFlags);

        if (ImGui::ArrowButton("TutorialGui_backButton", ImGuiDir_Left))
        {
            sendEvent(new GoBackEvent());
        }
        ImGui::SameLine(0.f, _spacingBackButtonToText);
        ImGui::Text(_resources.strings().tutorial());

        ImGui::End();

        ImGui::PopStyleColor(5);
        ImGui::PopFont();
        ImGui::PopStyleVar(1);
    }

    //button board window
    {
        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);
        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar |
                                       ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_AlwaysAutoResize |
                                       ImGuiWindowFlags_NoScrollbar |
                                       ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.00f, 0.00f, 0.00f, 0.00f));

        ImGui::Begin("TutorialGui_ButtonBoard", nullptr, windowFlags);

        if (ImGui::Button("##buttonLeft", ImVec2(50.f, 50.f)))
        {
        }

        if (ImGui::Button("##buttonRight", ImVec2(50.f, 50.f)))
        {
        }

        ImGui::PopStyleColor();

        ImGui::End();
    }
    //transparent content with dialog and left and right button
    {
        //button go left

        //button go right

        //dialog
    }

    popStyle();

    //ImGui::ShowMetricsWindow();
}

void TutorialGui::pushStyle()
{
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    //ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 20));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
}

void TutorialGui::popStyle()
{
    ImGui::PopStyleVar(6);
}

//void SelectionGui::pushStyle()
//{
//    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
//    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
//    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
//    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.f, 10.f));
//    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
//    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(_windowPadding, _windowPadding));
//    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(_buttonSpace, _buttonSpace));
//
//    ImGui::PushStyleColor(ImGuiCol_Button, _resources.style().buttonColorSelection);
//    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _resources.style().buttonColorSelection);
//    ImGui::PushStyleColor(ImGuiCol_ButtonActive, _resources.style().buttonColorPressedSelection);
//    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.00f, 0.00f, 0.00f, 0.00f));
//
//    if (_font)
//        ImGui::PushFont(_font);
//}
//
//void SelectionGui::popStyle()
//{
//    ImGui::PopStyleVar(7);
//    ImGui::PopStyleColor(4);
//
//    if (_font)
//        ImGui::PopFont();
//}
