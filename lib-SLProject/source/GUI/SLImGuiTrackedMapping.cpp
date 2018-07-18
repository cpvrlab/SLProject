//#############################################################################
//  File:      SLImGuiTrackedMapping.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <imgui.h>
#include <imgui_internal.h>

#include <SLImGuiTrackedMapping.h>
#include <SLCVTrackedMapping.h>
#include <SLCVMapStorage.h>

//-----------------------------------------------------------------------------
const char* SLImGuiTrackedMapping::_currItem = NULL;
int SLImGuiTrackedMapping::_currN = -1;
//-----------------------------------------------------------------------------
SLImGuiTrackedMapping::SLImGuiTrackedMapping(string name, SLCVTrackedMapping* mappingTracker)
    : SLImGuiInfosDialog(name),
    _mappingTracker(mappingTracker)
{
}
//-----------------------------------------------------------------------------
void SLImGuiTrackedMapping::buildInfos()
{
    if (ImGui::Button("Reset", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
        _mappingTracker->sm.requestStateIdle();
        while (!_mappingTracker->sm.hasStateIdle())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        _mappingTracker->Reset();
        _mappingTracker->sm.requestResume();
    }
    if (ImGui::Button("Bundle adjustment", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
        _mappingTracker->globalBundleAdjustment();
    }

    //add tracking state
    ImGui::Text("Tracking State : %s ", _mappingTracker->getPrintableState().c_str());
    //add number of matches map points in current frame
    ImGui::Text("Num Map Matches: %d ", _mappingTracker->getNMapMatches());
    //number of map points
    ImGui::Text("Num Map Pts: %d ", _mappingTracker->mapPointsCount());
    //add number of keyframes
    ImGui::Text("Number of Keyframes : %d ", _mappingTracker->getNumKeyFrames());
    //add loop closings counter
    ImGui::Text("Number of LoopClosings : %d ", _mappingTracker->getNumLoopClosings());

    SLCVKeyFrame* kf = _mappingTracker->currentKeyFrame();
    if (kf)
        ImGui::Text("Last keyframe : %d ", kf->mnId);

#ifdef ANDROID
    float bHeigth = 200.0f;
#else
    float bHeigth = 60.0f;
#endif

    if (ImGui::Button("Add key frame", ImVec2(ImGui::GetContentRegionAvailWidth(), bHeigth))) {
        _mappingTracker->mapNextFrame();
    }
}
