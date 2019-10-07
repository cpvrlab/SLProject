#include <AppDemoGuiVideoControls.h>
#include <AppWAI.h>

AppDemoGuiVideoControls::AppDemoGuiVideoControls(const std::string& name, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
{
}

void AppDemoGuiVideoControls::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Video controls", _activator, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("sdfsfsfsfsfdsfdsfdsfds");
    if (ImGui::Button("Pause", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        WAIApp::pauseVideo != WAIApp::pauseVideo;
    }

    ImGui::End();
}
