#include "GuiUtils.h"
#include <CVImage.h>
#include <Utils.h>

//add this to enable + operator on ImRect
#ifndef IMGUI_DEFINE_MATH_OPERATORS
#    define IMGUI_DEFINE_MATH_OPERATORS
#endif
#include <imgui_internal.h>

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

GLuint loadTexture(std::string fileName, bool flipX, bool flipY, float targetWdivH, int& cropW, int& cropH, int& textureW, int& textureH)
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
        textureW = image.width();
        textureH = image.height();

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
    {
        std::stringstream ss;
        ss << "imagePath does not exist: " << fileName;
        Utils::warnMsg("loadTexture", ss.str().c_str(), __LINE__, __FILE__);
    }

    return id;
}

GLuint loadTexture(std::string fileName, bool flipX, bool flipY, float targetWdivH)
{
    int cropW, cropH, textureW, textureH; //unused in this case
    return loadTexture(fileName, flipX, flipY, targetWdivH, cropW, cropH, textureW, textureH);
}

void deleteTexture(GLuint& id)
{
    if (id)
    {
        glDeleteTextures(1, &id);
        id = 0;
    }
}

std::vector<ImVec2> rotatePts(const std::vector<ImVec2>& pts, float angleDeg)
{
    std::vector<ImVec2> rotPts;

    return rotPts;
}

ImRect calcBoundingBox(const std::vector<ImVec2>& pts)
{
    float xMin, xMax, yMin, yMax;
    xMin = yMin = std::numeric_limits<float>::max();
    xMax = yMax = std::numeric_limits<float>::min();
    for (const ImVec2& pt : pts)
    {
        if (pt.x < xMin)
            xMin = pt.x;
        if (pt.x > xMax)
            xMax = pt.x;
        if (pt.y < yMin)
            yMin = pt.y;
        if (pt.y > yMax)
            yMax = pt.y;
    }

    return ImRect(ImVec2(xMin, xMax), ImVec2(xMax, yMax));
}

bool PoseShapeButton(const char* label, const ImVec2& size_arg, const ImVec4& col_normal, const ImVec4& col_active)
{
    ImGuiButtonFlags flags  = 0;
    ImGuiWindow*     window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return false;

    ImGuiContext&     g          = *GImGui;
    const ImGuiStyle& style      = g.Style;
    const ImGuiID     id         = window->GetID(label);
    const ImVec2      label_size = ImGui::CalcTextSize(label, NULL, true);

    ImVec2 pos = window->DC.CursorPos;
    if ((flags & ImGuiButtonFlags_AlignTextBaseLine) && style.FramePadding.y < window->DC.CurrLineTextBaseOffset) // Try to vertically align buttons that are smaller/have no padding so that text baseline matches (bit hacky, since it shouldn't be a flag)
        pos.y += window->DC.CurrLineTextBaseOffset - style.FramePadding.y;
    ImVec2 size = ImGui::CalcItemSize(size_arg, label_size.x + style.FramePadding.x * 2.0f, label_size.y + style.FramePadding.y * 2.0f);

    const ImRect bb(pos, pos + size);
    ImGui::ItemSize(size, style.FramePadding.y);
    if (!ImGui::ItemAdd(bb, id))
        return false;

    if (window->DC.ItemFlags & ImGuiItemFlags_ButtonRepeat)
        flags |= ImGuiButtonFlags_Repeat;
    bool hovered, held;
    bool pressed = ImGui::ButtonBehavior(bb, id, &hovered, &held, flags);

    // Render
    const ImU32 col = ImGui::GetColorU32((held && hovered) ? ImGuiCol_ButtonActive : hovered ? ImGuiCol_ButtonHovered : ImGuiCol_Button);
    ImGui::RenderNavHighlight(bb, id);
    ImGui::RenderFrame(bb.Min, bb.Max, col, true, style.FrameRounding);
    //RenderTextClipped(bb.Min + style.FramePadding, bb.Max - style.FramePadding, label, NULL, &label_size, style.ButtonTextAlign, &bb);

    ImVec2 c(bb.Min.x, 0.5f * (bb.Max.y + bb.Min.y));
    if (held && hovered)
    {
        window->DrawList->AddTriangleFilled(c, bb.Max, ImVec2(bb.Max.x, bb.Min.y), ImGui::GetColorU32(col_active));
        window->DrawList->AddCircleFilled(c, 10, ImGui::GetColorU32(col_active));
    }
    else
    {
        window->DrawList->AddTriangleFilled(c, bb.Max, ImVec2(bb.Max.x, bb.Min.y), ImGui::GetColorU32(col_normal));
        window->DrawList->AddCircleFilled(c, 10, ImGui::GetColorU32(col_normal));
    }

    return pressed;
}
};
