#include <TutorialGui.h>
#include <imgui_internal.h>
#include <CVImage.h>
#include <GuiUtils.h>

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
    _textureBackgroundId1 = loadTexture(texturePath + "earth2048_C.jpg", false, true, (float)screenWidthPix / (float)screenHeightPix);
    _textureBackgroundId2 = loadTexture(texturePath + "earthCloud1024_C.jpg", false, true, (float)screenWidthPix / (float)screenHeightPix);
    _currentBackgroundId  = _textureBackgroundId1;

    //load icon texture
    _textureIconLeftId  = loadTexture(texturePath + "icon_back.png", false, false, 1.f);
    _textureIconRightId = loadTexture(texturePath + "icon_back.png", true, false, 1.f);

    _textureIconBackWhiteId = loadTexture(texturePath + "icons/back1white.png", false, false, 1.f);
    _textureIconBackGrayId  = loadTexture(texturePath + "icons/back1gray.png", false, false, 1.f);
}

TutorialGui::~TutorialGui()
{
    deleteTexture(_textureBackgroundId1);
    deleteTexture(_textureBackgroundId2);
    deleteTexture(_textureIconLeftId);
    deleteTexture(_textureIconRightId);
    _currentBackgroundId = 0;
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
    //{
    //    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar;

    //    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    //    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    //    ImGui::SetNextWindowSize(ImVec2(_screenW, _screenH), ImGuiCond_Always);
    //    ImGui::Begin("TutorialGui_BackgroundTexture", nullptr, windowFlags | ImGuiWindowFlags_NoBringToFrontOnFocus);
    //    ImGui::Image((void*)(intptr_t)_currentBackgroundId, ImVec2(_screenW, _screenH));
    //    ImGui::End();

    //    ImGui::PopStyleVar(1);
    //}
    renderBackgroundTexture(_screenW, _screenH, _currentBackgroundId);
    pushStyle();

    float h = 10.0;
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
        h             = _context->FontSize + _context->Style.FramePadding.y * 2.0f; //same as ImGui::GetFrameHeight()
        float spacing = 0.5f * (_headerBarH - h);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(spacing, spacing));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(_buttonRounding, _buttonRounding));

        ImGui::Begin("TutorialGui_header", nullptr, windowFlags);

        //if (ImGui::ArrowButton("TutorialGui_backButton", ImGuiDir_Left))
        //{
        //    sendEvent(new GoBackEvent());
        //}
        //if (ImGui::ImageButton((ImTextureID)_textureIconLeftId, ImVec2(h, h)))
        if (ImGui::ImageButton((ImTextureID)_textureIconBackWhiteId, (ImTextureID)_textureIconBackGrayId, ImVec2(h, h)))
        {
            sendEvent(new GoBackEvent());
        }

        ImGui::SameLine(0.f, _spacingBackButtonToText);
        ImGui::Text(_resources.strings().tutorial());

        ImGui::End();

        ImGui::PopStyleColor(5);
        ImGui::PopFont();
        ImGui::PopStyleVar(3);
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
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(_buttonRounding, _buttonRounding));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.00f, 0.00f, 0.00f, 0.00f));
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.00f, 0.00f, 0.00f, 0.00f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.00f, 0.00f, 0.00f, 0.00f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.f, 1.f, 1.f, 0.5f));

        ImGui::Begin("TutorialGui_ButtonBoard", nullptr, windowFlags);

        if (ImGui::ImageButton((ImTextureID)_textureIconLeftId, ImVec2(h, h)))
        {
            if (_currentBackgroundId == _textureBackgroundId1)
                _currentBackgroundId = _textureBackgroundId2;
            else if (_currentBackgroundId == _textureBackgroundId2)
                _currentBackgroundId = _textureBackgroundId1;
        }

        if (ImGui::ImageButton((ImTextureID)_textureIconRightId, ImVec2(h, h)))
        {
            if (_currentBackgroundId == _textureBackgroundId1)
                _currentBackgroundId = _textureBackgroundId2;
            else if (_currentBackgroundId == _textureBackgroundId2)
                _currentBackgroundId = _textureBackgroundId1;
        }

        //if (ImGui::ImageButton((ImTextureID)_textureIconRightId, (ImTextureID)_textureIconLeftId, ImVec2(h, h)))
        //{
        //    if (_currentBackgroundId == _textureBackgroundId1)
        //        _currentBackgroundId = _textureBackgroundId2;
        //    else if (_currentBackgroundId == _textureBackgroundId2)
        //        _currentBackgroundId = _textureBackgroundId1;
        //}

        ImGui::End();
        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar(2);
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
