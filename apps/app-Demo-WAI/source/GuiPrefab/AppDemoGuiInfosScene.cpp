#include <imgui.h>
#include <imgui_internal.h>

#include <SLApplication.h>
#include <CVCapture.h>
#include <AppDemoGuiInfosScene.h>
//-----------------------------------------------------------------------------
AppDemoGuiInfosScene::AppDemoGuiInfosScene(string name, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
{
}

//-----------------------------------------------------------------------------
void AppDemoGuiInfosScene::buildInfos(SLScene* s, SLSceneView* sv)
{
    // Calculate window position for dynamic status bar at the bottom of the main window
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoTitleBar;
    window_flags |= ImGuiWindowFlags_NoResize;
    SLfloat  w    = (SLfloat)sv->scrW();
    ImVec2   size = ImGui::CalcTextSize(s->info().c_str(), nullptr, true, w);
    SLfloat  h    = size.y + SLGLImGui::fontPropDots * 1.2f;
    SLstring info = "Scene Info: " + s->info();

    ImGui::SetNextWindowPos(ImVec2(0, sv->scrH() - h));
    ImGui::SetNextWindowSize(ImVec2(w, h));
    ImGui::Begin("Scene Information", _activator, window_flags);
    ImGui::TextWrapped("%s", info.c_str());
    ImGui::End();
}

