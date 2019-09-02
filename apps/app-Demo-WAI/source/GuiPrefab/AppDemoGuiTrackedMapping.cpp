#include <imgui.h>
#include <imgui_internal.h>

#include <AppDemoGuiTrackedMapping.h>

//-----------------------------------------------------------------------------
const char* AppDemoGuiTrackedMapping::_currItem = NULL;
int         AppDemoGuiTrackedMapping::_currN    = -1;
//-----------------------------------------------------------------------------
AppDemoGuiTrackedMapping::AppDemoGuiTrackedMapping(string name, WAI::ModeOrbSlam2* orbSlamMode, bool* activator)
  : AppDemoGuiInfosDialog(name, activator),
    _orbSlamMode(orbSlamMode)
{
}
//-----------------------------------------------------------------------------
void AppDemoGuiTrackedMapping::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Tracked Mapping", _activator, ImGuiWindowFlags_AlwaysAutoResize);
    if (ImGui::Button("Reset", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        _orbSlamMode->reset();
    }

    //add tracking state
    ImGui::Text("Tracking State : %s ", _orbSlamMode->getPrintableState().c_str());
    //add number of matches map points in current frame
    ImGui::Text("Num Map Matches: %d ", _orbSlamMode->getMapPointMatchesCount());
    //number of map points
    ImGui::Text("Num Map Pts: %d ", _orbSlamMode->getMapPointCount());
    //add number of keyframes
    ImGui::Text("Number of Keyframes : %d ", _orbSlamMode->getKeyFrameCount());
    ImGui::InputInt("Min. frames before next KF", &_orbSlamMode->mMinFrames, 5, 0);

    //add loop closings counter
    ImGui::Text("Number of LoopClosings : %d ", _orbSlamMode->getLoopCloseCount());
    //ImGui::Text("Number of LoopClosings : %d ", _wai->mpLoopCloser->numOfLoopClosings());
    ImGui::Text("Loop closing status : %s ", _orbSlamMode->getLoopCloseStatus().c_str());
    ImGui::Text("Keyframes in Loop closing queue : %d", _orbSlamMode->getKeyFramesInLoopCloseQueueCount());

    //show 2D key points in video image
    bool b = _orbSlamMode->getTrackOptFlow();
    if (ImGui::Checkbox("Track optical flow", &b))
    {
        _orbSlamMode->setTrackOptFlow(b);
    }

    ImGui::End();
}
