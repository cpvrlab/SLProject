#include "GuiUtils.h"
#include <CVImage.h>
#include <Utils.h>

namespace ErlebAR
{
void renderBackgroundTexture(float screenW, float screenH, GLuint texId)
{
    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar |
                                   ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoScrollbar |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(screenW, screenH), ImGuiCond_Always);
    ImGui::Begin("TutorialGui_BackgroundTexture", nullptr, windowFlags);
    ImGui::Image((void*)(intptr_t)texId, ImVec2(screenW, screenH));
    ImGui::End();

    ImGui::PopStyleVar(1);
}

void renderHeaderBar(std::string               id,
                     float                     width,
                     float                     height,
                     const ImVec4&             backgroundColor,
                     const ImVec4&             textColor,
                     const ImVec4&             buttonColor,
                     const ImVec4&             buttonColorPressed,
                     ImFont*                   font,
                     float                     buttonRounding,
                     float                     buttonHeight,
                     GLuint                    texId,
                     float                     spacingButtonToText,
                     const char*               text,
                     std::function<void(void)> cb)
{
    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar |
                                   ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoScrollbar;

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_Always);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, backgroundColor);
    ImGui::PushStyleColor(ImGuiCol_Text, textColor);
    ImGui::PushStyleColor(ImGuiCol_Button, buttonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, buttonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, buttonColorPressed);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));                          //we adjust this by SetNextWindowPos for child windows
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(buttonRounding, buttonRounding)); //distance button border to button texture
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, buttonRounding);                        //here same as framepadding, frame padding needs minimum the rounding size
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::PushFont(font);

    ImGui::Begin((id + "_header").c_str(), nullptr, windowFlags);

    float buttonWinPadding = 0.5f * (height - buttonHeight);
    float texSize          = buttonHeight - 2 * buttonRounding;
    ImGui::SetNextWindowPos(ImVec2(buttonWinPadding, buttonWinPadding), ImGuiCond_Always);
    ImGui::BeginChild((id + "_button").c_str(), ImVec2(buttonHeight, buttonHeight), false, windowFlags);
    //if (ImGui::ImageButton((ImTextureID)texId, (ImTextureID)texIdPressed, ImVec2(texSize, texSize)))
    if (ImGui::ImageButton((ImTextureID)texId, ImVec2(texSize, texSize)))
    {
        cb();
    }
    ImGui::EndChild();

    //change window padding so that text is centered
    float textWinPadding = 0.5f * (height - font->FontSize);
    ImGui::SetNextWindowPos(ImVec2(buttonWinPadding + buttonHeight + spacingButtonToText, textWinPadding), ImGuiCond_Always);
    ImGui::BeginChild((id + "_text").c_str());
    ImGui::Text(text);
    ImGui::EndChild();

    ImGui::End();

    ImGui::PopFont();
    ImGui::PopStyleVar(7);
    ImGui::PopStyleColor(5);
}

void renderAreaPlaceButtons(std::map<AreaId, Area> areas)
{
}

GLuint loadTexture(std::string fileName, bool flipX, bool flipY, float targetWdivH, int& cropW, int& cropH)
{
    GLuint id = 0;

    if (Utils::fileExists(fileName))
    {
        // load texture image
        CVImage image(fileName);
        if (flipX)
            image.flipX();
        if (flipY)
            image.flipY();
        //crop image to screen size
        image.crop(targetWdivH, cropW, cropH);

        // Create a OpenGL texture identifier
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);

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
        Utils::warnMsg("loadTexture", "imagePath does not exist!", __LINE__, __FILE__);

    return id;
}

GLuint loadTexture(std::string fileName, bool flipX, bool flipY, float targetWdivH)
{
    int cropW, cropH;
    return loadTexture(fileName, flipX, flipY, targetWdivH, cropW, cropH);
}

void deleteTexture(GLuint& id)
{
    if (id)
    {
        glDeleteTextures(1, &id);
        id = 0;
    }
}
};
