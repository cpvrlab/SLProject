#include <SelectionGui.h>
#include <ErlebAR.h>
#include <imgui_internal.h>
#include <CVImage.h>
#include <float.h>

SelectionGui::SelectionGui(sm::EventHandler& eventHandler,
                           int               dotsPerInch,
                           int               screenWidthPix,
                           int               screenHeightPix,
                           std::string       fontPath,
                           std::string       texturePath)
  : sm::EventSender(eventHandler)
{
    resize(screenWidthPix, screenHeightPix);
    int fontHeightDots = _buttonSz.y * ErlebAR::ButtonTextH;

    //add font and store index
    SLstring DroidSans = fontPath + "Roboto-Medium.ttf";
    if (Utils::fileExists(DroidSans))
    {
        _font = _context->IO.Fonts->AddFontFromFileTTF(DroidSans.c_str(), fontHeightDots);
    }
    else
    {
        Utils::warnMsg("SelectionGui", "SelectionGui: font does not exist!", __LINE__, __FILE__);
    }

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
        Utils::warnMsg("SelectionGui", "imagePath does not exist!", __LINE__, __FILE__);
}

SelectionGui::~SelectionGui()
{
    if (_textureBackgroundId)
    {
        glDeleteTextures(1, &_textureBackgroundId);
        _textureBackgroundId = 0;
    }
}

void SelectionGui::onResize(SLint scrW, SLint scrH)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH);
}

void SelectionGui::resize(int scrW, int scrH)
{
    _screenWPix = (float)scrW;
    _screenHPix = (float)scrH;

    _windowPadding = 0.f;
    _buttonSpace   = 0.02f * _screenHPix;

    _buttonRounding         = ErlebAR::ButtonRounding * _screenHPix;
    float frameButtonBoardB = 0.1f * _screenHPix;
    float frameButtonBoardR = 0.1f * _screenWPix;
    _buttonBoardW           = 0.5f * _screenWPix;
    _buttonBoardH           = 0.6f * _screenHPix;
    _buttonBoardPosX        = _screenWPix - _buttonBoardW - frameButtonBoardR;
    _buttonBoardPosY        = _screenHPix - _buttonBoardH - frameButtonBoardB;

    int nButVert = 8; //number of buttons in vertical direction
    int buttonH  = (_buttonBoardH - 2 * _windowPadding - (nButVert - 1) * _buttonSpace) / nButVert;
    _buttonSz    = {-FLT_MIN, (float)buttonH};
}

void SelectionGui::pushStyle()
{
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(_windowPadding, _windowPadding));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(_buttonSpace, _buttonSpace));

    ImGui::PushStyleColor(ImGuiCol_Button, ErlebAR::SelectionButtonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ErlebAR::SelectionButtonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ErlebAR::SelectionButtonPressedColor);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.00f, 0.00f, 0.00f, 0.00f));

    if (_font)
        ImGui::PushFont(_font);
}

void SelectionGui::popStyle()
{
    ImGui::PopStyleVar(7);
    ImGui::PopStyleColor(4);

    if (_font)
        ImGui::PopFont();
}

void SelectionGui::build(SLScene* s, SLSceneView* sv)
{
    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    //background texture window
    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenWPix, _screenHPix), ImGuiCond_Always);
        ImGui::Begin("SelectionGui_BackgroundTexture", nullptr, windowFlags | ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::Image((void*)(intptr_t)_textureBackgroundId, ImVec2(_screenWPix, _screenHPix));
        ImGui::End();
    }
    ImGui::PopStyleVar(1);

    //push styles at first
    pushStyle();
    //button board window
    {
        ImGui::SetNextWindowPos(ImVec2(_buttonBoardPosX, _buttonBoardPosY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_buttonBoardW, _buttonBoardH), ImGuiCond_Always);
        ImGui::Begin("SelectionGui_ButtonBoard", nullptr, windowFlags);

        if (ImGui::Button("Test", _buttonSz))
        {
            sendEvent(new StartTestEvent());
        }

        if (ImGui::Button("Camera Test", _buttonSz))
        {
            sendEvent(new StartCameraTestEvent());
        }

        if (ImGui::Button("Avanches", _buttonSz))
        {
            sendEvent(new StartErlebarEvent(Location::AVANCHES));
        }

        if (ImGui::Button("Augst", _buttonSz))
        {
            sendEvent(new StartErlebarEvent(Location::AUGST));
        }

        if (ImGui::Button("Christoffel", _buttonSz))
        {
            sendEvent(new StartErlebarEvent(Location::CHRISTOFFEL));
        }

        if (ImGui::Button("Biel", _buttonSz))
        {
            sendEvent(new StartErlebarEvent(Location::BIEL));
        }

        if (ImGui::Button("Settings", _buttonSz))
        {
            sendEvent(new ShowSettingsEvent());
        }

        if (ImGui::Button("About", _buttonSz))
        {
            sendEvent(new ShowAboutEvent());
        }

        ImGui::End();
    }

    popStyle();
}
