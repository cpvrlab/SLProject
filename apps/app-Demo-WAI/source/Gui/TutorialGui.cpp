#include <TutorialGui.h>
#include <imgui_internal.h>
#include <CVImage.h>
#include <GuiUtils.h>
#include <SLVec2.h>
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
    renderBackgroundTexture(_screenW, _screenH, _currentBackgroundId);
    //header bar
    renderHeaderBar("TutorialGui",
                    _screenW,
                    _headerBarH,
                    _resources.style().headerBarBackgroundTranspColor,
                    _resources.style().headerBarTextColor,
                    _resources.style().headerBarBackButtonColor,
                    _resources.style().headerBarBackButtonPressedColor,
                    _fontBig,
                    _buttonRounding,
                    _headerBarH * 0.8,
                    _textureIconBackWhiteId,
                    _textureIconBackGrayId,
                    _spacingBackButtonToText,
                    _resources.strings().tutorial(),
                    [&]() { sendEvent(new GoBackEvent()); });

    //button board window
    //float buttonSize,
    //float winPosStartY,
    //
    {
        float buttonSize = _headerBarH * 0.8f;

        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar |
                                       ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_AlwaysAutoResize |
                                       ImGuiWindowFlags_NoScrollbar |
                                       ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.00f, 0.00f, 0.00f, 0.00f));
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.00f, 0.00f, 0.00f, 0.00f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.00f, 0.00f, 0.00f, 0.00f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.f, 1.f, 1.f, 0.5f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(_buttonRounding, _buttonRounding));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

        ImGui::Begin("TutorialGui_ButtonBoard", nullptr, windowFlags);

        //mGui::PushID()
        if (ImGui::ImageButton((ImTextureID)_textureIconLeftId, ImVec2(buttonSize, buttonSize)))
        {
            if (_currentBackgroundId == _textureBackgroundId1)
                _currentBackgroundId = _textureBackgroundId2;
            else if (_currentBackgroundId == _textureBackgroundId2)
                _currentBackgroundId = _textureBackgroundId1;
        }

        if (ImGui::ImageButton((ImTextureID)_textureIconRightId, ImVec2(buttonSize, buttonSize)))
        {
            if (_currentBackgroundId == _textureBackgroundId1)
                _currentBackgroundId = _textureBackgroundId2;
            else if (_currentBackgroundId == _textureBackgroundId2)
                _currentBackgroundId = _textureBackgroundId1;
        }

        ImGui::End();

        ImGui::PopStyleVar(6);
        ImGui::PopStyleColor(4);
    }

    //popStyle();
    //ImGui::PopStyleVar(4);

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
