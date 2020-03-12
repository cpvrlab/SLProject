#include <SelectionGui.h>
#include <SLScene.h>
#include <SLSceneView.h>

SelectionGui::SelectionGui(int dotsPerInch, std::string fontPath)
{
    _pixPerMM = (float)dotsPerInch / 25.4f;
    // Scale for proportional and fixed size fonts
    float dpiScaleProp  = dotsPerInch / 120.0f;
    float dpiScaleFixed = dotsPerInch / 142.0f;

    // Default settings for the first time
    float fontPropDots  = std::max(16.0f * dpiScaleProp, 16.0f);
    float fontFixedDots = std::max(13.0f * dpiScaleFixed, 13.0f);

    //load fonts
    loadFonts(fontPropDots, fontFixedDots, fontPath);
}

void setStyleColors(ImGuiStyle* style)
{
    ImVec4* colors = style->Colors;

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
    static float  frameSize          = 0.f * _pixPerMM; //frame between dialog and window
    static int    nButHoriz          = 2;               //number of buttons in horizontal direction
    static int    nButVert           = 3;               //number of buttons in vertical direction
    static float  buttonSpace        = 5.f * _pixPerMM; //space between buttons
    static float  windowPadding      = 7.f * _pixPerMM; //space l, r, b, t between window and buttons (window padding left does not work as expected)
    static ImVec4 buttonColor        = (ImVec4)ImColor(0.98 * 255, 0.647 * 255, 0 * 255, 0.8 * 255);
    static ImVec4 buttonColorPressed = (ImVec4)ImColor(0.937f * 255.f, 0.945f * 255.f, 0.953 * 255.f, 0.8f * 255.f);

    //calculate resulting sizes:
    float  dialogW = sv->scrW() - 2 * frameSize;
    float  dialogH = sv->scrH() - 2 * frameSize;
    int    buttonW = (dialogW - 2 * windowPadding - (nButHoriz - 1) * buttonSpace) / nButHoriz;
    int    buttonH = (dialogH - 2 * windowPadding - (nButVert - 1) * buttonSpace) / nButVert;
    ImVec2 buttonSz(buttonW, buttonH);

    ImGuiStyle& style     = ImGui::GetStyle();
    style.FrameRounding   = 0.0f; //0.0f, 12.0f
    style.FrameBorderSize = 0.f;
    style.FramePadding    = ImVec2(0.f, 0.f);
    style.FrameRounding   = 0.f;
    style.WindowRounding  = 0.f;
    style.WindowPadding   = ImVec2(0, windowPadding);         //space l, r, b, t between window and buttons (window padding left does not work as expected)
    style.ItemSpacing     = ImVec2(buttonSpace, buttonSpace); //space between buttons

    setStyleColors(&style);

    {
        ImGui::SetNextWindowPos(ImVec2(frameSize, frameSize), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(dialogW, dialogH), ImGuiCond_Always);

        ImGui::Begin("", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::PushStyleColor(ImGuiCol_Button, buttonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, buttonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, buttonColorPressed);

        ImGui::NewLine();
        ImGui::SameLine(windowPadding);
        if (ImGui::Button("Test", buttonSz))
        {
            _selection = AppMode::TEST;
        }
        ImGui::SameLine();
        if (ImGui::Button("Camera Test", buttonSz))
        {
            _selection = AppMode::CAMERA_TEST;
        }

        ImGui::NewLine();
        ImGui::SameLine(windowPadding);
        if (ImGui::Button("Avanches", buttonSz))
        {
            _selection = AppMode::AVANCHES;
        }
        ImGui::SameLine();
        if (ImGui::Button("Augst", buttonSz))
        {
            _selection = AppMode::AUGST;
        }

        ImGui::NewLine();
        ImGui::SameLine(windowPadding);
        if (ImGui::Button("Christoffel", buttonSz)) // Buttons return true when clicked (most widgets return true when edited/activated)
        {
            _selection = AppMode::CHRISTOFFELTOWER;
        }
        ImGui::SameLine(/*buttonSpace + buttonW + buttonSpace*/);
        if (ImGui::Button("Biel", buttonSz))
        {
            _selection = AppMode::BIEL;
        }

        ImGui::PopStyleColor(3);

        ImGui::End();
    }
}
