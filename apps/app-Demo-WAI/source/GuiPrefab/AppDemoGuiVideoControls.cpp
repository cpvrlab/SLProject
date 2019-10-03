#include <AppDemoGuiVideoControls.h>
#include <AppWAI.h>

AppDemoGuiVideoControls::AppDemoGuiVideoControls(const std::string& name, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
{
}

void AppDemoGuiVideoControls::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Video controls", _activator, 0);
    ImGui::Separator();
    if (ImGui::Button("Pause", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        WAIApp::pauseVideo != WAIApp::pauseVideo;
    }

    if (ImGui::Button("Save current frame with candidates", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        if (WAIApp::mode != nullptr && WAIApp::mode->retainImage())
        {
            WAIFrame frame = WAIApp::mode->getCurrentFrame();
            WAIKeyFrame * ref = frame.mpReferenceKF;

            //Save image from frame and its reference keyframe
        }
    }

    ImGui::End();
}
