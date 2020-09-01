#if defined(__clang__)
#    pragma GCC diagnostic ignored "-Wint-to-void-pointer-cast"
#endif

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

void renderPanningBackgroundTexture(float x, float y, float w, float h, float screenW, float screenH, GLuint texId)
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
    ImGui::Image((void*)(intptr_t)texId, ImVec2(screenW, screenH), ImVec2(x, y), ImVec2(x + w, y + h));
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
                     std::function<void(void)> cb,
                     float                     opacity)
{
    //if headerbar is transparent, we disable it
    if (opacity < 0.0001f)
        return;

    //reduce opacity of colors (for invisible
    const ImVec4 backgroundColorOp    = {backgroundColor.x, backgroundColor.y, backgroundColor.z, backgroundColor.w * opacity};
    const ImVec4 textColorOp          = {textColor.x, textColor.y, textColor.z, textColor.w * opacity};
    const ImVec4 buttonColorOp        = {buttonColor.x, buttonColor.y, buttonColor.z, buttonColor.w * opacity};
    const ImVec4 buttonColorPressedOp = {buttonColorPressed.x, buttonColorPressed.y, buttonColorPressed.z, buttonColorPressed.w * opacity};
    const ImVec4 btnImgCol            = {1.f, 1.f, 1.f, opacity};

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar |
                                   ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoScrollbar;

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_Always);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, backgroundColorOp);
    ImGui::PushStyleColor(ImGuiCol_Text, textColorOp);
    ImGui::PushStyleColor(ImGuiCol_Button, buttonColorOp);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, buttonColorOp);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, buttonColorPressedOp);
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
    if (ImGui::ImageButton((ImTextureID)texId, ImVec2(texSize, texSize), ImVec2(0, 0), ImVec2(1, 1), -1, ImVec4(0, 0, 0, 0), btnImgCol))
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

GLuint loadTexture(std::string fileName, bool flipX, bool flipY, float targetWdivH, int& textureW, int& textureH)
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

std::vector<ImVec2> rotatePts(const std::vector<ImVec2>& pts, float angleDeg, const ImVec2& c)
{
    float alphaRad = DEG2RAD * angleDeg;
    float cosAlpha = cos(alphaRad);
    float sinAlpha = sin(alphaRad);

    std::vector<ImVec2> rotPts(pts.size());
    for (int i = 0; i < pts.size(); ++i)
    {
        ImVec2&       rotPt = rotPts[i];
        const ImVec2& pt    = pts[i];

        //x2 = ((x1 - x0) * cos(a)) - ((y1 - y0) * sin(a)) + x0;
        rotPt.x = (pt.x - c.x) * cosAlpha - (pt.y - c.y) * sinAlpha + c.x;
        //y2 = ((x1 - x0) * sin(a)) + ((y1 - y0) * cos(a)) + y0;
        rotPt.y = (pt.x - c.x) * sinAlpha + (pt.y - c.y) * cosAlpha + c.y;

        ////x' = xcos(alpha) - ysin(alpha)
        //rotPt.x = pt.x * cosAlpha - pt.y * sinAlpha;
        ////y' = xsin(alpha) + ycos(alpha)
        //rotPt.y = pt.x * sinAlpha + pt.y * cosAlpha;
    }
    return rotPts;
}

ImRect calcBoundingBoxPts(const std::vector<ImVec2>& pts)
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

    return ImRect(ImVec2(xMin, yMin), ImVec2(xMax, yMax));
}

ImRect calcBoundingBox(const std::vector<ImVec2>& triPts, float r)
{
    const ImVec2&       c = triPts[0];
    std::vector<ImVec2> circle(4);
    circle[0] = ImVec2(c.x - r, c.y); //left
    circle[1] = ImVec2(c.x, c.y + r); //bottom
    circle[2] = ImVec2(c.x + r, c.y); //right
    circle[3] = ImVec2(c.x, c.y - r); //top

    circle.insert(circle.end(), triPts.begin(), triPts.end());
    return calcBoundingBoxPts(circle);
}

std::vector<ImVec2> constructTrianglePts(const float angleDeg, const ImVec2& center, const float length, const float width)
{
    std::vector<ImVec2> triangle(3);
    triangle[0] = ImVec2(center);                                     //root (point center)
    triangle[1] = ImVec2(center.x + length, center.y + 0.5f * width); //bottom right
    triangle[2] = ImVec2(center.x + length, center.y - 0.5f * width); //top right
    return rotatePts(triangle, angleDeg, triangle[0]);
}

bool poseShapeButton(const char*   label,
                     const ImVec2& sizeArg,
                     const float   circleRadius,
                     const float   viewTriangleLength,
                     const float   viewTriangleWidth,
                     const float   viewAngleDeg,
                     const ImVec4& colNormal,
                     const ImVec4& colActive)
{
    ImGuiButtonFlags flags  = 0;
    ImGuiWindow*     window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return false;

    ImGuiContext&     g     = *GImGui;
    const ImGuiStyle& style = g.Style;
    const ImGuiID     id    = window->GetID(label);
    //const ImVec2      label_size = ImGui::CalcTextSize(label, NULL, true);

    ImVec2 pos = window->DC.CursorPos;
    //construct pose icon
    std::vector<ImVec2> rotTriPts = constructTrianglePts(viewAngleDeg, pos, viewTriangleLength, viewTriangleWidth);
    ImRect              bb        = calcBoundingBox(rotTriPts, circleRadius);

    //if ((flags & ImGuiButtonFlags_AlignTextBaseLine) && style.FramePadding.y < window->DC.CurrLineTextBaseOffset) // Try to vertically align buttons that are smaller/have no padding so that text baseline matches (bit hacky, since it shouldn't be a flag)
    //    pos.y += window->DC.CurrLineTextBaseOffset - style.FramePadding.y;
    //ImVec2 size = ImGui::CalcItemSize(sizeArg, label_size.x + style.FramePadding.x * 2.0f, label_size.y + style.FramePadding.y * 2.0f);

    //const ImRect bb(pos, pos + size);
    ImGui::ItemSize(bb.GetSize(), style.FramePadding.y);
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

    //ImVec2 c(bb.Min.x, 0.5f * (bb.Max.y + bb.Min.y));
    if (held && hovered)
    {
        //window->DrawList->AddTriangleFilled(c, bb.Max, ImVec2(bb.Max.x, bb.Min.y), ImGui::GetColorU32(colActive));
        window->DrawList->AddTriangleFilled(rotTriPts[0], rotTriPts[1], rotTriPts[2], ImGui::GetColorU32(colActive));
        window->DrawList->AddCircleFilled(pos, circleRadius, ImGui::GetColorU32(colActive));
    }
    else
    {
        //window->DrawList->AddTriangleFilled(c, bb.Max, ImVec2(bb.Max.x, bb.Min.y), ImGui::GetColorU32(colNormal));
        window->DrawList->AddTriangleFilled(rotTriPts[0], rotTriPts[1], rotTriPts[2], ImGui::GetColorU32(colNormal));
        window->DrawList->AddCircleFilled(pos, circleRadius, ImGui::GetColorU32(colNormal));
    }

    return pressed;
}

void waitingSpinner(const char*   label,
                    const ImVec2& pos,
                    const float   indicatorRadius,
                    const ImVec4& mainColor,
                    const ImVec4& backdropColor,
                    const int     circleCount,
                    const float   speed)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
    {
        return;
    }

    ImGuiContext&     g     = *GImGui;
    const ImGuiID     id    = window->GetID(label);
    const ImGuiStyle& style = g.Style;

    const float  circle_radius = indicatorRadius / 10.0f;
    const ImRect bb(pos, ImVec2(pos.x + indicatorRadius * 2.0f, pos.y + indicatorRadius * 2.0f));
    ImGui::ItemSize(bb, style.FramePadding.y);
    if (!ImGui::ItemAdd(bb, id))
    {
        return;
    }
    const float t             = g.Time;
    const auto  degree_offset = 2.0f * IM_PI / circleCount;
    for (int i = 0; i < circleCount; ++i)
    {
        const float x      = indicatorRadius * std::sin(degree_offset * i);
        const float y      = indicatorRadius * std::cos(degree_offset * i);
        const float growth = std::max(0.0f, std::sin(t * speed - i * degree_offset));
        ImVec4      color;
        color.x = mainColor.x * growth + backdropColor.x * (1.0f - growth);
        color.y = mainColor.y * growth + backdropColor.y * (1.0f - growth);
        color.z = mainColor.z * growth + backdropColor.z * (1.0f - growth);
        color.w = mainColor.w * growth + backdropColor.w * (1.0f - growth);
        window->DrawList->AddCircleFilled(ImVec2(pos.x + indicatorRadius + x,
                                                 pos.y + indicatorRadius - y),
                                          circle_radius + growth * circle_radius,
                                          ImGui::GetColorU32(color));
    }
}
};
