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

void renderHeaderBar()
{
}

unsigned int loadTexture(std::string fileName, bool flipX, bool flipY, float targetWdivH)
{
    unsigned int id = 0;

    if (Utils::fileExists(fileName))
    {
        // load texture image
        CVImage image(fileName);
        if (flipX)
            image.flipX();
        if (flipY)
            image.flipY();
        //crop image to screen size
        image.crop(targetWdivH);

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

void deleteTexture(unsigned int& id)
{
    if (id)
    {
        glDeleteTextures(1, &id);
        id = 0;
    }
}
};
