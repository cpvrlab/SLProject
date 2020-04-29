#ifndef GUI_UTILS_H
#define GUI_UTILS_H

#include <string>
#include <functional>
#include <imgui.h>
//SLGLState to correctly include opengl
#include <SLGLState.h>
#include <ErlebAR.h>

namespace ErlebAR
{
void renderBackgroundTexture(float screenW, float screenH, GLuint texId);
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
                     std::function<void(void)> cb);

GLuint loadTexture(std::string fileName, bool flipX, bool flipY, float targetWdivH, int& cropW, int& cropH, int& textureW, int& textureH);
GLuint loadTexture(std::string fileName, bool flipX, bool flipY, float targetWdivH);

void deleteTexture(GLuint& id);
bool PoseShapeButton(const char*   label,
                     const ImVec2& sizeArg,
                     const float   circleRadius,
                     const float   viewTriangleLength,
                     const float   viewTriangleWidth,
                     const float   viewAngleDeg,
                     const ImVec4& colNormal,
                     const ImVec4& colActive);
};

enum class GuiAlignment
{
    TOP_LEFT,
    TOP_RIGHT,
    BOTTOM_LEFT,
    BOTTOM_RIGHT
};

class BackButton
{
public:
    using Callback = std::function<void(void)>;

    BackButton(int          dotsPerInch,
               int          screenWidthPix,
               int          screenHeightPix,
               GuiAlignment alignment,
               float        distFrameHorizMM,
               float        distFrameVertMM,
               ImVec2       buttonSizeMM,
               Callback     pressedCB,
               ImFont*      font)
      : _alignment(alignment),
        _pressedCB(pressedCB),
        _font(font)
    {
        float pixPerMM = (float)dotsPerInch / 25.4f;

        _buttonSizePix = {buttonSizeMM.x * pixPerMM, buttonSizeMM.y * pixPerMM};
        _windowSizePix = {_buttonSizePix.x + 2 * _windowPadding, _buttonSizePix.y + 2 * _windowPadding};

        //top
        if (_alignment == GuiAlignment::TOP_LEFT || _alignment == GuiAlignment::TOP_RIGHT)
        {
            _windowPos.y = distFrameVertMM * pixPerMM;
        }
        else //bottom
        {
            _windowPos.y = screenHeightPix - distFrameVertMM * pixPerMM - _windowSizePix.y;
        }

        //left
        if (_alignment == GuiAlignment::BOTTOM_LEFT || _alignment == GuiAlignment::TOP_LEFT)
        {
            _windowPos.x = distFrameHorizMM * pixPerMM;
        }
        else //right
        {
            _windowPos.x = screenWidthPix - distFrameHorizMM * pixPerMM - _windowSizePix.x;
        }
    }
    BackButton()
    {
    }

    void render()
    {
        {
            ImGuiStyle& style   = ImGui::GetStyle();
            style.WindowPadding = ImVec2(0, _windowPadding); //space l, r, b, t between window and buttons (window padding left does not work as expected)

            //back button
            ImGui::SetNextWindowPos(_windowPos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(_windowSizePix, ImGuiCond_Always);

            ImGui::Begin("AppDemoWaiGui", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::PushStyleColor(ImGuiCol_Button, _buttonColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _buttonColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, _buttonColorPressed);
            if (_font)
                ImGui::PushFont(_font);
            ImGui::NewLine();
            ImGui::SameLine(_windowPadding);
            if (ImGui::Button("back", _buttonSizePix))
            {
                if (_pressedCB)
                    _pressedCB();
            }

            ImGui::PopStyleColor(3);
            if (_font)
                ImGui::PopFont();

            ImGui::End();
        }
    }

private:
    ImVec4 _buttonColor = {BFHColors::OrangePrimary.r,
                           BFHColors::OrangePrimary.g,
                           BFHColors::OrangePrimary.b,
                           BFHColors::OrangePrimary.a};

    ImVec4 _buttonColorPressed = {BFHColors::GrayLogo.r,
                                  BFHColors::GrayLogo.g,
                                  BFHColors::GrayLogo.b,
                                  BFHColors::GrayLogo.a};

    GuiAlignment _alignment;
    ImVec2       _windowPos;
    //calculated sized of dialogue
    ImVec2 _windowSizePix;
    //button size (in pixel)
    ImVec2 _buttonSizePix;
    //distance between button border and window border
    int _windowPadding = 2.f;

    Callback _pressedCB = nullptr;

    ImFont* _font = nullptr;
};

#endif //GUI_UTILS_H
