#include "LogWindow.h"
#include <imgui_internal.h>

LogWindow::LogWindow(int screenWidth, int screenHeight, ImFont* fontHeading, ImFont* fontText)
  : _screenW(screenWidth),
    _screenH(screenHeight),
    _fontHeading(fontHeading),
    _fontText(fontText)
{
    _autoScroll = true;
    clear();
}

void LogWindow::clear()
{
    _buf.clear();
    _lineOffsets.clear();
    _lineOffsets.push_back(0);
}

void LogWindow::post(const std::string& message)
{
    addLog(message.c_str());
}

void LogWindow::draw(const char* title, bool* p_open)
{

    float framePadding = 0.02f * _screenH;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(framePadding, framePadding));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(framePadding, framePadding));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(framePadding, framePadding));
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 2.f * framePadding + _fontHeading->FontSize);

    ImGui::PushFont(_fontHeading);
    if (!ImGui::Begin(title, p_open))
    {
        ImGui::End();
        ImGui::PopStyleVar(5);
        ImGui::PopFont();
        return;
    }
    // pop heading font
    ImGui::PopFont();
    ImGui::PushFont(_fontText);

    // Options menu
    if (ImGui::BeginPopup("Options"))
    {
        ImGui::Checkbox("Auto-scroll", &_autoScroll);
        ImGui::EndPopup();
    }

    // Main window
    if (ImGui::Button("Options"))
        ImGui::OpenPopup("Options");
    ImGui::SameLine();
    bool clearIt = ImGui::Button("Clear");
    ImGui::SameLine();
    bool copy = ImGui::Button("Copy");
    ImGui::SameLine();
    _filter.Draw("Filter", -100.0f);

    ImGui::Separator();
    ImGui::BeginChild("scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

    if (clearIt)
        clear();
    if (copy)
        ImGui::LogToClipboard();

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    const char* buf     = _buf.begin();
    const char* buf_end = _buf.end();
    if (_filter.IsActive())
    {
        // In this example we don't use the clipper when _filter is enabled.
        // This is because we don't have a random access on the result on our filter.
        // A real application processing logs with ten of thousands of entries may want to store the result of search/filter.
        // especially if the filtering function is not trivial (e.g. reg-exp).
        for (int line_no = 0; line_no < _lineOffsets.Size; line_no++)
        {
            const char* line_start = buf + _lineOffsets[line_no];
            const char* line_end   = (line_no + 1 < _lineOffsets.Size) ? (buf + _lineOffsets[line_no + 1] - 1) : buf_end;
            if (_filter.PassFilter(line_start, line_end))
                ImGui::TextUnformatted(line_start, line_end);
        }
    }
    else
    {
        // The simplest and easy way to display the entire buffer:
        //   ImGui::TextUnformatted(buf_begin, buf_end);
        // And it'll just work. TextUnformatted() has specialization for large blob of text and will fast-forward to skip non-visible lines.
        // Here we instead demonstrate using the clipper to only process lines that are within the visible area.
        // If you have tens of thousands of items and their processing cost is non-negligible, coarse clipping them on your side is recommended.
        // Using ImGuiListClipper requires A) random access into your data, and B) items all being the  same height,
        // both of which we can handle since we an array pointing to the beginning of each line of text.
        // When using the filter (in the block of code above) we don't have random access into the data to display anymore, which is why we don't use the clipper.
        // Storing or skimming through the search result would make it possible (and would be recommended if you want to search through tens of thousands of entries)
        ImGuiListClipper clipper;
        clipper.Begin(_lineOffsets.Size);
        while (clipper.Step())
        {
            for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++)
            {
                const char* line_start = buf + _lineOffsets[line_no];
                const char* line_end   = (line_no + 1 < _lineOffsets.Size) ? (buf + _lineOffsets[line_no + 1] - 1) : buf_end;
                ImGui::TextUnformatted(line_start, line_end);
            }
        }
        clipper.End();
    }
    ImGui::PopStyleVar();

    if (_autoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
        ImGui::SetScrollHereY(1.0f);

    ImGui::EndChild();
    ImGui::End();

    ImGui::PopStyleVar(5);
    ImGui::PopFont();
}

void LogWindow::addLog(const char* fmt, ...) IM_FMTARGS(2)
{
    int     old_size = _buf.size();
    va_list args;
    va_start(args, fmt);
    _buf.appendfv(fmt, args);
    va_end(args);
    for (int new_size = _buf.size(); old_size < new_size; old_size++)
        if (_buf[old_size] == '\n')
            _lineOffsets.push_back(old_size + 1);
}
