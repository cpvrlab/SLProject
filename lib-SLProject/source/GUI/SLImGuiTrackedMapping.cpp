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
        //_mappingTracker->Pause();
        _mappingTracker->sm.requestStateIdle();
        while (!_mappingTracker->sm.hasStateIdle())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        _mappingTracker->Reset();
        //_mappingTracker->Resume();
        _mappingTracker->sm.requestResume();
    }
    if (ImGui::Button("Bundle adjustment", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
        _mappingTracker->globalBundleAdjustment();
    }

    //add tracking state
    ImGui::Text("Tracking State : %s ", _mappingTracke->getPrintableState().c_str());
    //add number of matches map points in current frame
    ImGui::Text("Num Map Matches: %d ", _mappingTracker->getNMapMatches());
    //number of map points
    ImGui::Text("Num Map Pts: %d ", _mappingTracker->mapPointsCount());
    //add number of keyframes
    ImGui::Text("Number of Keyframes : %d ", _mappingTracker->getNumKeyFrames());
    ImGui::InputInt("Min. frames before next KF", &_mappingTracker->mMinFrames, 5, 0);

    if (_mappingTracker->mpLoopCloser)
    {
        //add loop closings counter
        ImGui::Text("Number of LoopClosings : %d ", _mappingTracker->mpLoopCloser->numOfLoopClosings());
        ImGui::Text("Loop closing status : %s ", _mappingTracker->mpLoopCloser->getStatusString());
        ImGui::Text("Keyframes in Loop closing queue : %d", _mappingTracker->mpLoopCloser->numOfKfsInQueue());

#if 0
        ImGui::Text("Number of Loop Candidates : %d", _mappingTracker->mpLoopCloser->numOfCandidates());
        ImGui::Text("Number of Consistent Candidates : %d", _mappingTracker->mpLoopCloser->numOfConsistentCandidates());
        ImGui::Text("Number of Consistent Groups : %d", _mappingTracker->mpLoopCloser->numOfConsistentGroups());
#endif
    }

    SLCVKeyFrame* kf = _mappingTracker->currentKeyFrame();
    if (kf)
    {
        ImGui::Text("Last keyframe : %d ", kf->mnId);
    }
    else
    {
        ImGui::Text("No keyframe yet");
    }

#ifdef ANDROID
    float bHeigth = 200.0f;
#else
    float bHeigth = 60.0f;
#endif

    if (ImGui::Button("Attempt loop close", ImVec2(ImGui::GetContentRegionAvailWidth(), bHeigth))) {
        _mappingTracker->mpLoopCloser->startLoopCloseAttempt();
    }
}
