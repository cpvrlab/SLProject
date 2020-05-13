#include <SelectionGui.h>
#include <ErlebAR.h>
#include <imgui_internal.h>
#include <CVImage.h>
#include <float.h>
#include <ErlebAREvents.h>

using namespace ErlebAR;

SelectionGui::SelectionGui(const ImGuiEngine&  imGuiEngine,
                           sm::EventHandler&   eventHandler,
                           ErlebAR::Resources& resources,
                           int                 dotsPerInch,
                           int                 screenWidthPix,
                           int                 screenHeightPix,
                           std::string         fontPath,
                           std::string         texturePath)
  : ImGuiWrapper(imGuiEngine.context(), imGuiEngine.renderer()),
    sm::EventSender(eventHandler),
    _resources(resources)
{
    resize(screenWidthPix, screenHeightPix);

    //load background texture
    std::string imagePath = texturePath + "earth2048_C.jpg";
    if (Utils::fileExists(imagePath))
    {
        // load texture image
        CVImage image(imagePath);
        image.flipY();
        //crop image to screen size
        int cropW, cropH;
        image.crop((float)screenWidthPix / (float)screenHeightPix, cropW, cropH);

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

void SelectionGui::onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH, scr2fbX, scr2fbY);
}

void SelectionGui::resize(int scrW, int scrH)
{
    _screenWPix = (float)scrW;
    _screenHPix = (float)scrH;

    _windowPadding = 0.f;
    _framePadding  = 0.02f * _screenHPix;
    _buttonSpace   = 0.02f * _screenHPix;

    _buttonRounding         = _resources.style().buttonRounding * _screenHPix;
    float frameButtonBoardB = 0.1f * _screenHPix;
    float frameButtonBoardR = 0.1f * _screenWPix;
    _buttonBoardW           = 0.5f * _screenWPix;
    _buttonBoardH           = 0.6f * _screenHPix;
    _buttonBoardPosX        = _screenWPix - _buttonBoardW - frameButtonBoardR;
    _buttonBoardPosY        = _screenHPix - _buttonBoardH - frameButtonBoardB;

    int nButVert = 6; //number of buttons in vertical direction
    int buttonH  = (int)((_buttonBoardH - 2 * _windowPadding - (nButVert - 1) * _buttonSpace) / nButVert);
    _buttonSz    = {-FLT_MIN, (float)buttonH};
}

void SelectionGui::pushStyle()
{
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(_framePadding, _framePadding));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(_windowPadding, _windowPadding));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(_buttonSpace, _buttonSpace));

    ImGui::PushStyleColor(ImGuiCol_Button, _resources.style().buttonColorSelection);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _resources.style().buttonColorSelection);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, _resources.style().buttonColorPressedSelection);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.00f, 0.00f, 0.00f, 0.00f));

    ImGui::PushFont(_resources.fonts().selectBtns);
}

void SelectionGui::popStyle()
{
    ImGui::PopStyleVar(7);
    ImGui::PopStyleColor(4);

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

        if (ImGui::Button("Avenches", _buttonSz))
        {
            sendEvent(new StartErlebarEvent("SelectionGui", LocationId::AVENCHES));
        }

        if (ImGui::Button("Augst", _buttonSz))
        {
            sendEvent(new StartErlebarEvent("SelectionGui", LocationId::AUGST));
        }

        if (ImGui::Button("Christoffel", _buttonSz))
        {
            sendEvent(new StartErlebarEvent("SelectionGui", LocationId::CHRISTOFFEL));
        }

        if (ImGui::Button(_resources.strings().tutorial(), _buttonSz))
        {
            sendEvent(new StartTutorialEvent("SelectionGui"));
        }

        if (ImGui::Button(_resources.strings().settings(), _buttonSz))
        {
            sendEvent(new ShowSettingsEvent("SelectionGui"));
        }

        if (ImGui::Button(_resources.strings().about(), _buttonSz))
        {
            sendEvent(new ShowAboutEvent("SelectionGui"));
        }

        ImGui::End();
    }

    //developer button board
    if (_resources.developerMode)
    {
        ImGui::SetNextWindowPos(ImVec2(0, _buttonBoardPosY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenWPix - _buttonBoardW - 0.1f * _screenWPix, _buttonBoardH), ImGuiCond_Always);
        ImGui::Begin("SelectionGui_ButtonBoardDevelMode", nullptr, windowFlags);

        ImVec2 develButtonSize(-FLT_MIN /*_screenWPix - _buttonSz.x*/, _buttonSz.y);
        if (ImGui::Button("Test", develButtonSize))
        {
            sendEvent(new StartTestEvent("SelectionGui"));
        }

        if (ImGui::Button("Camera Test", develButtonSize))
        {
            sendEvent(new StartCameraTestEvent("SelectionGui"));
        }

        if (ImGui::Button("Biel", develButtonSize))
        {
            sendEvent(new StartErlebarEvent("SelectionGui", LocationId::BIEL));
        }

        ImGui::End();
    }
    popStyle();

    //debug: draw log window
    _resources.logWinDraw();
}
