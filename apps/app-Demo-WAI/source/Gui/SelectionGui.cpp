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
}

void SelectionGui::popStyle()
{
    ImGui::PopStyleVar(7);
    ImGui::PopStyleColor(4);
}

void SelectionGui::setStyleColors()
{
    ImVec4* colors = _guiStyle.Colors;

    colors[ImGuiCol_Text]                 = ImVec4(0.90f, 0.90f, 0.90f, 1.00f);
    colors[ImGuiCol_TextDisabled]         = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
    colors[ImGuiCol_WindowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_ChildBg]              = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]              = ImVec4(0.11f, 0.11f, 0.14f, 0.92f);
    colors[ImGuiCol_Border]               = ImVec4(0.50f, 0.50f, 0.50f, 0.50f);
    colors[ImGuiCol_BorderShadow]         = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]              = ImVec4(0.43f, 0.43f, 0.43f, 0.39f);
    colors[ImGuiCol_FrameBgHovered]       = ImVec4(0.47f, 0.47f, 0.69f, 0.40f);
    colors[ImGuiCol_FrameBgActive]        = ImVec4(0.42f, 0.41f, 0.64f, 0.69f);
    colors[ImGuiCol_TitleBg]              = ImVec4(0.27f, 0.27f, 0.54f, 0.83f);
    colors[ImGuiCol_TitleBgActive]        = ImVec4(0.32f, 0.32f, 0.63f, 0.87f);
    colors[ImGuiCol_TitleBgCollapsed]     = ImVec4(0.40f, 0.40f, 0.80f, 0.20f);
    colors[ImGuiCol_MenuBarBg]            = ImVec4(0.40f, 0.40f, 0.55f, 0.80f);
    colors[ImGuiCol_ScrollbarBg]          = ImVec4(0.20f, 0.25f, 0.30f, 0.60f);
    colors[ImGuiCol_ScrollbarGrab]        = ImVec4(0.40f, 0.40f, 0.80f, 0.30f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.40f, 0.40f, 0.80f, 0.40f);
    colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.41f, 0.39f, 0.80f, 0.60f);
    colors[ImGuiCol_CheckMark]            = ImVec4(0.90f, 0.90f, 0.90f, 0.50f);
    colors[ImGuiCol_SliderGrab]           = ImVec4(1.00f, 1.00f, 1.00f, 0.30f);
    colors[ImGuiCol_SliderGrabActive]     = ImVec4(0.41f, 0.39f, 0.80f, 0.60f);
    colors[ImGuiCol_Button]               = ImVec4(0.35f, 0.40f, 0.61f, 0.62f);
    colors[ImGuiCol_ButtonHovered]        = ImVec4(0.40f, 0.48f, 0.71f, 0.79f);
    colors[ImGuiCol_ButtonActive]         = ImVec4(0.46f, 0.54f, 0.80f, 1.00f);
    colors[ImGuiCol_Header]               = ImVec4(0.40f, 0.40f, 0.90f, 0.45f);
    colors[ImGuiCol_HeaderHovered]        = ImVec4(0.45f, 0.45f, 0.90f, 0.80f);
    colors[ImGuiCol_HeaderActive]         = ImVec4(0.53f, 0.53f, 0.87f, 0.80f);
    colors[ImGuiCol_Separator]            = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_SeparatorHovered]     = ImVec4(0.60f, 0.60f, 0.70f, 1.00f);
    colors[ImGuiCol_SeparatorActive]      = ImVec4(0.70f, 0.70f, 0.90f, 1.00f);
    colors[ImGuiCol_ResizeGrip]           = ImVec4(1.00f, 1.00f, 1.00f, 0.16f);
    colors[ImGuiCol_ResizeGripHovered]    = ImVec4(0.78f, 0.82f, 1.00f, 0.60f);
    colors[ImGuiCol_ResizeGripActive]     = ImVec4(0.78f, 0.82f, 1.00f, 0.90f);
    colors[ImGuiCol_CloseButton]          = ImVec4(0.50f, 0.50f, 0.90f, 0.50f);
    colors[ImGuiCol_CloseButtonHovered]   = ImVec4(0.70f, 0.70f, 0.90f, 0.60f);
    colors[ImGuiCol_CloseButtonActive]    = ImVec4(0.70f, 0.70f, 0.70f, 1.00f);
    colors[ImGuiCol_PlotLines]            = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered]     = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogram]        = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
    colors[ImGuiCol_TextSelectedBg]       = ImVec4(0.00f, 0.00f, 1.00f, 0.35f);
    colors[ImGuiCol_ModalWindowDarkening] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);
    colors[ImGuiCol_DragDropTarget]       = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
}

void SelectionGui::build(SLScene* s, SLSceneView* sv)
{
    pushStyle();
    {
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
    }

    popStyle();
}
