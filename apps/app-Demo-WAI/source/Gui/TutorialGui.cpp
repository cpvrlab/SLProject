#include <TutorialGui.h>
#include <imgui_internal.h>
#include <CVImage.h>
#include <GuiUtils.h>
#include <SLVec2.h>
#include <ErlebAREvents.h>

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
    int cropW, cropH;
    _textureBackgroundId1 = loadTexture(texturePath + "earth2048_C.jpg", false, true, (float)screenWidthPix / (float)screenHeightPix);
    _textureBackgroundId2 = loadTexture(texturePath + "earthCloud1024_C.jpg", false, true, (float)screenWidthPix / (float)screenHeightPix);
    _currentBackgroundId  = _textureBackgroundId1;

    //load icon texture
    _textureIconLeftId  = loadTexture(texturePath + "left1white.png", false, false, 1.f);
    _textureIconRightId = loadTexture(texturePath + "left1white.png", true, false, 1.f);
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
    float buttonSize = _headerBarH * _resources.style().headerBarButtonH;
    renderHeaderBar("TutorialGui",
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
                    _resources.strings().tutorial(),
                    [&]() { sendEvent(new GoBackEvent()); });

    //button board window
    {
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

        //left button
        float buttonWinPadding = 0.5f * (_headerBarH - buttonSize); //same as for header bar in which case it depends on the header bar height!
        float texSize          = buttonSize - 2 * _buttonRounding;
        float buttonYPos       = (_screenH - buttonSize) * 0.5f;
        ImGui::SetNextWindowPos(ImVec2(buttonWinPadding, buttonYPos), ImGuiCond_Always);
        ImGui::BeginChild("TutorialGui_winButtonLeft", ImVec2(buttonSize, buttonSize));
        ImGui::PushID("TutorialGui_imgButtonLeft");
        if (ImGui::ImageButton((ImTextureID)_textureIconLeftId, ImVec2(texSize, texSize)))
        {
            if (_currentBackgroundId == _textureBackgroundId1)
                _currentBackgroundId = _textureBackgroundId2;
            else if (_currentBackgroundId == _textureBackgroundId2)
                _currentBackgroundId = _textureBackgroundId1;
        }
        ImGui::PopID();
        ImGui::EndChild();

        //right button
        float buttonRightXPos = _screenW - buttonWinPadding - buttonSize;
        ImGui::SetNextWindowPos(ImVec2(buttonRightXPos, buttonYPos), ImGuiCond_Always);
        ImGui::BeginChild("TutorialGui_winButtonRight", ImVec2(buttonSize, buttonSize));
        ImGui::PushID("TutorialGui_imgButtonRight");
        if (ImGui::ImageButton((ImTextureID)_textureIconRightId, ImVec2(texSize, texSize)))
        {
            if (_currentBackgroundId == _textureBackgroundId1)
                _currentBackgroundId = _textureBackgroundId2;
            else if (_currentBackgroundId == _textureBackgroundId2)
                _currentBackgroundId = _textureBackgroundId1;
        }
        ImGui::PopID();
        ImGui::EndChild();

        ImGui::End();

        ImGui::PopStyleVar(6);
        ImGui::PopStyleColor(4);
    }

    //ImGui::ShowMetricsWindow();
}
