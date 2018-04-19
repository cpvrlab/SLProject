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
        _mappingTracker->Reset();
        //_mappingTracker->setState(SLCVTrackedMapping::INITIALIZE);
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


    //else if (ImGui::Button("Track VO", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
    //    _mappingTracker->setState(SLCVTrackedMapping::TRACK_VO);
    //}
    //else if (ImGui::Button("Track 3D Pts", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
    //    _mappingTracker->setState(SLCVTrackedMapping::TRACK_3DPTS);
    //}
    //else if (ImGui::Button("Track optical flow", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
    //    _mappingTracker->setState(SLCVTrackedMapping::TRACK_OPTICAL_FLOW);
    //}

#ifdef ANDROID
    float bHeigth = 200.0f;
#else
    float bHeigth = 60.0f;
#endif

    if (ImGui::Button("Add key frame", ImVec2(ImGui::GetContentRegionAvailWidth(), bHeigth))) {
        _mappingTracker->mapNextFrame();
    }

    //ImGui::Separator();
    //if (ImGui::Button("Save map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
    //    _mappingTracker->saveMap();
    //}
    //ImGui::Separator();
    //if (ImGui::Button("New map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
    //    _mappingTracker->saveMap();
    //}
    //ImGui::Separator();

    //{
    //    if (ImGui::BeginCombo("Current", _currItem)) // The second parameter is the label previewed before opening the combo.
    //    {
    //        for (int n = 0; n < SLCVMapStorage::existingMapNames.size(); n++)
    //        {
    //            bool isSelected = (_currItem == SLCVMapStorage::existingMapNames[n].c_str()); // You can store your selection however you want, outside or inside your objects
    //            if (ImGui::Selectable(SLCVMapStorage::existingMapNames[n].c_str(), isSelected)) {
    //                _currItem = SLCVMapStorage::existingMapNames[n].c_str();
    //                _currN = n;
    //            }
    //            if (isSelected)
    //                ImGui::SetItemDefaultFocus();   // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
    //        }
    //        ImGui::EndCombo();
    //    }
    //}
    //if (ImGui::Button("Load map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
    //    _mappingTracker->saveMap();
    //}
}
