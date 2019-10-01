#include <imgui.h>
#include <imgui_internal.h>

#include <AppWAI.h>
#include <AppDemoGuiTrackedMapping.h>

//-----------------------------------------------------------------------------
const char* AppDemoGuiTrackedMapping::_currItem = NULL;
int         AppDemoGuiTrackedMapping::_currN    = -1;
//-----------------------------------------------------------------------------
AppDemoGuiTrackedMapping::AppDemoGuiTrackedMapping(string name, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
{
}
//-----------------------------------------------------------------------------
void AppDemoGuiTrackedMapping::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Tracked Mapping", _activator, ImGuiWindowFlags_AlwaysAutoResize);
    if (ImGui::Button("Reset", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        WAIApp::mode->reset();
    }

    if (ImGui::Button("Disable Mapping", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        WAIApp::mode->disableMapping();
    }
    if (ImGui::Button("Enable Mapping", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        WAIApp::mode->enableMapping();
    }

    //add tracking state
    ImGui::Text("Tracking State : %s ", WAIApp::mode->getPrintableState().c_str());
    //add number of matches map points in current frame
    ImGui::Text("Num Map Matches: %d ", WAIApp::mode->getMapPointMatchesCount());
    //number of map points
    ImGui::Text("Num Map Pts: %d ", WAIApp::mode->getMapPointCount());
    //add number of keyframes
    ImGui::Text("Number of Keyframes : %d ", WAIApp::mode->getKeyFrameCount());
    ImGui::InputInt("Min. frames before next KF", &WAIApp::mode->mMinFrames, 5, 0);

    //add loop closings counter
    ImGui::Text("Number of LoopClosings : %d ", WAIApp::mode->getLoopCloseCount());
    //ImGui::Text("Number of LoopClosings : %d ", _wai->mpLoopCloser->numOfLoopClosings());
    ImGui::Text("Loop closing status : %s ", WAIApp::mode->getLoopCloseStatus().c_str());
    ImGui::Text("Keyframes in Loop closing queue : %d", WAIApp::mode->getKeyFramesInLoopCloseQueueCount());

    //show 2D key points in video image
    bool b = WAIApp::mode->getTrackOptFlow();
    if (ImGui::Checkbox("Track optical flow", &b))
    {
        WAIApp::mode->setTrackOptFlow(b);
    }

    ImGui::End();
}
