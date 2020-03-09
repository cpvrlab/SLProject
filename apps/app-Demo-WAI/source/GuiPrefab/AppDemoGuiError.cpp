#include <imgui.h>
#include <imgui_internal.h>

#include <AppDemoGuiError.h>

#include <utility>
#include <SLGLImGui.h>
//-----------------------------------------------------------------------------
AppDemoGuiError::AppDemoGuiError(string name, bool* activator)
  : AppDemoGuiInfosDialog(std::move(name), activator)
{
}

//-----------------------------------------------------------------------------
void AppDemoGuiError::buildInfos(SLScene* s, SLSceneView* sv)
{
    // Calculate window position for dynamic status bar at the bottom of the main window
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoTitleBar;
    window_flags |= ImGuiWindowFlags_NoResize;
    SLfloat w    = (SLfloat)sv->viewportW() - 10;
    ImVec2  size = ImGui::CalcTextSize(_errorMsg.c_str(), nullptr, true, w);
    SLfloat h    = size.y + SLGLImGui::fontPropDots * 1.2f + 20;

    ImGui::SetNextWindowPos(ImVec2(5, (sv->scrH() * 0.5f) - (h * 0.5f)));
    ImGui::SetNextWindowSize(ImVec2(w, h));

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
    ImGui::Begin("Error", _activator, window_flags);

    ImGui::TextWrapped("%s", _errorMsg.c_str());

    ImGui::End();
    ImGui::PopStyleColor();
}
