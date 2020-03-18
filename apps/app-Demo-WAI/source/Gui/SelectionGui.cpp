#include <SelectionGui.h>
#include <SLScene.h>
#include <SLSceneView.h>

SelectionGui::SelectionGui(int         dotsPerInch,
                           int         screenWidthPix,
                           int         screenHeightPix,
                           std::string fontPath)
{
    _pixPerMM = (float)dotsPerInch / 25.4f;

    _windowPadding = 7.f * _pixPerMM;
    _buttonSpace   = 5.f * _pixPerMM;

    _buttonColor = {BFHColors::OrangePrimary.r,
                    BFHColors::OrangePrimary.g,
                    BFHColors::OrangePrimary.b,
                    BFHColors::OrangePrimary.a};

    _buttonColorPressed = {BFHColors::GrayLogo.r,
                           BFHColors::GrayLogo.g,
                           BFHColors::GrayLogo.b,
                           BFHColors::GrayLogo.a};

    int nButHoriz = 2; //number of buttons in horizontal direction
    int nButVert  = 3; //number of buttons in vertical direction

    _frameSizePix = 0.f * _pixPerMM; //frame between dialog and window

    //calculate resulting sizes:
    _dialogW    = screenWidthPix - 2 * _frameSizePix;
    _dialogH    = screenHeightPix - 2 * _frameSizePix;
    int buttonW = (_dialogW - 2 * _windowPadding - (nButHoriz - 1) * _buttonSpace) / nButHoriz;
    int buttonH = (_dialogH - 2 * _windowPadding - (nButVert - 1) * _buttonSpace) / nButVert;
    _buttonSz   = {(float)buttonW, (float)buttonH};

    //add font and store index
    float    fontHeightMM   = 5.f;
    int      fontHeightDots = fontHeightMM * _pixPerMM;
    SLstring DroidSans      = fontPath + "DroidSans.ttf";
    if (Utils::fileExists(DroidSans))
    {
        ImGuiIO& io = ImGui::GetIO();
        _font       = io.Fonts->AddFontFromFileTTF(DroidSans.c_str(), fontHeightDots);
    }
    else
    {
        Utils::warnMsg("SelectionGui", "SelectionGui: font does not exist!", __LINE__, __FILE__);
    }
}

void SelectionGui::pushStyle()
{
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, _windowPadding));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(_buttonSpace, _buttonSpace));

    ImGui::PushStyleColor(ImGuiCol_Button, _buttonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _buttonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, _buttonColorPressed);
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
    //push styles at first
    pushStyle();

    ImGui::SetNextWindowPos(ImVec2(_frameSizePix, _frameSizePix), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(_dialogW, _dialogH), ImGuiCond_Always);

    ImGui::Begin("", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::NewLine();
    ImGui::SameLine(_windowPadding);
    if (ImGui::Button("Test", _buttonSz))
    {
        _selection = AppMode::TEST;
    }
    ImGui::SameLine();
    if (ImGui::Button("Camera Test", _buttonSz))
    {
        _selection = AppMode::CAMERA_TEST;
    }

    ImGui::NewLine();
    ImGui::SameLine(_windowPadding);
    if (ImGui::Button("Avanches", _buttonSz))
    {
        _selection = AppMode::AVANCHES;
    }
    ImGui::SameLine();
    if (ImGui::Button("Augst", _buttonSz))
    {
        _selection = AppMode::AUGST;
    }

    ImGui::NewLine();
    ImGui::SameLine(_windowPadding);
    if (ImGui::Button("Christoffel", _buttonSz))
    {
        _selection = AppMode::CHRISTOFFELTOWER;
    }
    ImGui::SameLine();
    if (ImGui::Button("Biel", _buttonSz))
    {
        _selection = AppMode::BIEL;
    }

    ImGui::End();

    popStyle();
}
