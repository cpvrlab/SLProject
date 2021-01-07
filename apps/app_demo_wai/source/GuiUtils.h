#ifndef GUI_UTILS_H
#define GUI_UTILS_H

#include <string>
#include <functional>
#include <imgui.h>
#include <imgui_internal.h>
//SLGLState to correctly include opengl
#include <SLGLState.h>
#include <ErlebAR.h>

namespace ErlebAR
{

void renderPanningBackgroundTexture(float x, float y, float w, float h, float screenW, float screenH, GLuint texId);
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
                     std::function<void(void)> cb,
                     float                     opacity = 1.f);

GLuint loadTexture(std::string fileName, bool flipX, bool flipY, float targetWdivH, int& cropW, int& cropH, int& textureW, int& textureH);
GLuint loadTexture(std::string fileName, bool flipX, bool flipY, float targetWdivH, int& textureW, int& textureH);
GLuint loadTexture(std::string fileName, bool flipX, bool flipY, float targetWdivH);
GLuint generateTexture(int& textureW, int& textureH);

void deleteTexture(GLuint& id);
bool poseShapeButton(const char*   label,
                     const ImVec2& sizeArg,
                     const float   circleRadius,
                     const float   viewTriangleLength,
                     const float   viewTriangleWidth,
                     const float   viewAngleDeg,
                     const ImVec4& colNormal,
                     const ImVec4& colActive,
                     const bool    pressable);

void waitingSpinner(const char*   label,
                    const ImVec2& pos,
                    const float   indicatorRadius,
                    const ImVec4& mainColor,
                    const ImVec4& backdropColor,
                    const int     circleCount,
                    const float   speed);
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
            style.WindowPadding = ImVec2(0, (float)_windowPadding); //space l, r, b, t between window and buttons (window padding left does not work as expected)

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
            ImGui::SameLine((float)_windowPadding);
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
    ImVec4 _buttonColor = {BFHColors::OrangeLogo.r,
                           BFHColors::OrangeLogo.g,
                           BFHColors::OrangeLogo.b,
                           BFHColors::OrangeLogo.a};

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
    int _windowPadding = 2;

    Callback _pressedCB = nullptr;

    ImFont* _font = nullptr;
};

//class LogWindow : public CustomLog
//{
//public:
//    LogWindow(int screenWidth, int screenHeight)
//      : _screenW(screenWidth),
//        _screenH(screenHeight)
//    {
//        _autoScroll = true;
//        clear();
//    }
//
//    void clear()
//    {
//        _buf.clear();
//        _lineOffsets.clear();
//        _lineOffsets.push_back(0);
//    }
//
//    void post(const std::string& message) override
//    {
//        addLog(message.c_str());
//    }
//
//    void draw(ImFont* font, const char* title, bool* p_open = NULL)
//    {
//        //ImGui::PushFont(font);
//        float         framePadding     = 0.02f * _screenH;
//        ImGuiContext* c                = ImGui::GetCurrentContext();
//        float         targetFontHeight = 0.035 * _screenH;
//        c->Font->Scale                 = targetFontHeight / c->Font->FontSize;
//        //c->Font->Scale             = 1.5f;
//        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(framePadding, framePadding));
//        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
//        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(framePadding, framePadding));
//        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(framePadding, framePadding));
//        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 2.f * framePadding + c->Font->FontSize);
//
//        if (!ImGui::Begin(title, p_open))
//        {
//            ImGui::End();
//            ImGui::PopStyleVar(5);
//            //ImGui::PopFont();
//            return;
//        }
//
//        // Options menu
//        if (ImGui::BeginPopup("Options"))
//        {
//            ImGui::Checkbox("Auto-scroll", &_autoScroll);
//            ImGui::EndPopup();
//        }
//
//        // Main window
//        if (ImGui::Button("Options"))
//            ImGui::OpenPopup("Options");
//        ImGui::SameLine();
//        bool clearIt = ImGui::Button("Clear");
//        ImGui::SameLine();
//        bool copy = ImGui::Button("Copy");
//        ImGui::SameLine();
//        _filter.Draw("Filter", -100.0f);
//
//        ImGui::Separator();
//        ImGui::BeginChild("scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
//
//        if (clearIt)
//            clear();
//        if (copy)
//            ImGui::LogToClipboard();
//
//        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
//        const char* buf     = _buf.begin();
//        const char* buf_end = _buf.end();
//        if (_filter.IsActive())
//        {
//            // In this example we don't use the clipper when _filter is enabled.
//            // This is because we don't have a random access on the result on our filter.
//            // A real application processing logs with ten of thousands of entries may want to store the result of search/filter.
//            // especially if the filtering function is not trivial (e.g. reg-exp).
//            for (int line_no = 0; line_no < _lineOffsets.Size; line_no++)
//            {
//                const char* line_start = buf + _lineOffsets[line_no];
//                const char* line_end   = (line_no + 1 < _lineOffsets.Size) ? (buf + _lineOffsets[line_no + 1] - 1) : buf_end;
//                if (_filter.PassFilter(line_start, line_end))
//                    ImGui::TextUnformatted(line_start, line_end);
//            }
//        }
//        else
//        {
//            // The simplest and easy way to display the entire buffer:
//            //   ImGui::TextUnformatted(buf_begin, buf_end);
//            // And it'll just work. TextUnformatted() has specialization for large blob of text and will fast-forward to skip non-visible lines.
//            // Here we instead demonstrate using the clipper to only process lines that are within the visible area.
//            // If you have tens of thousands of items and their processing cost is non-negligible, coarse clipping them on your side is recommended.
//            // Using ImGuiListClipper requires A) random access into your data, and B) items all being the  same height,
//            // both of which we can handle since we an array pointing to the beginning of each line of text.
//            // When using the filter (in the block of code above) we don't have random access into the data to display anymore, which is why we don't use the clipper.
//            // Storing or skimming through the search result would make it possible (and would be recommended if you want to search through tens of thousands of entries)
//            ImGuiListClipper clipper;
//            clipper.Begin(_lineOffsets.Size);
//            while (clipper.Step())
//            {
//                for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++)
//                {
//                    const char* line_start = buf + _lineOffsets[line_no];
//                    const char* line_end   = (line_no + 1 < _lineOffsets.Size) ? (buf + _lineOffsets[line_no + 1] - 1) : buf_end;
//                    ImGui::TextUnformatted(line_start, line_end);
//                }
//            }
//            clipper.End();
//        }
//        ImGui::PopStyleVar();
//
//        if (_autoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
//            ImGui::SetScrollHereY(1.0f);
//
//        ImGui::EndChild();
//        ImGui::End();
//
//        ImGui::PopStyleVar(5);
//        //ImGui::PopFont();
//    }
//
//private:
//    void addLog(const char* fmt, ...) IM_FMTARGS(2)
//    {
//        int     old_size = _buf.size();
//        va_list args;
//        va_start(args, fmt);
//        _buf.appendfv(fmt, args);
//        va_end(args);
//        for (int new_size = _buf.size(); old_size < new_size; old_size++)
//            if (_buf[old_size] == '\n')
//                _lineOffsets.push_back(old_size + 1);
//    }
//
//    ImGuiTextBuffer _buf;
//    ImGuiTextFilter _filter;
//    ImVector<int>   _lineOffsets; // Index to lines offset. We maintain this with AddLog() calls, allowing us to have a random access on lines
//    bool            _autoScroll;  // Keep scrolling if already at the bottom
//
//    int _screenW;
//    int _screenH;
//};

#endif //GUI_UTILS_H
