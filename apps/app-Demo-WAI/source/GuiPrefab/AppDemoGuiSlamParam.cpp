//#############################################################################
//  File:      AppDemoGuiTestBenchOpen.cpp
//  Author:    Luc Girod
//  Date:      September 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <imgui.h>
#include <imgui_internal.h>

#include <Utils.h>
#include <OrbSlam/KPextractor.h>
#include <OrbSlam/SURFextractor.h>
#include <OrbSlam/ORBextractor.h>
#include <OrbSlam/ORBmatcher.h>
#include <AppDemoGuiSlamParam.h>
#include <Utils.h>
#include <CVCapture.h>
#include <GLSLextractor.h>
#include <WAIApp.h>
//-----------------------------------------------------------------------------
AppDemoGuiSlamParam::AppDemoGuiSlamParam(const std::string&              name,
                                         bool*                           activator,
                                         std::queue<WAIEvent*>*          eventQueue,
                                         const std::vector<std::string>& extractorIdToNames)
  : AppDemoGuiInfosDialog(name, activator),
    _eventQueue(eventQueue),
    _extractorIdToNames(extractorIdToNames)
{
    _currentId       = 2;
    _iniCurrentId    = 1;
    _markerCurrentId = 1;
}

void AppDemoGuiSlamParam::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Slam Param", _activator, ImGuiWindowFlags_AlwaysAutoResize);
    if (ImGui::BeginCombo("Extractor", _extractorIdToNames.at(_currentId).c_str()))
    {
        for (int i = 0; i < _extractorIdToNames.size(); i++)
        {
            bool isSelected = (_currentId == i); // You can store your selection however you want, outside or inside your objects
            if (ImGui::Selectable(_extractorIdToNames.at(i).c_str(), isSelected))
                _currentId = i;
            if (isSelected)
                ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
        }
        ImGui::EndCombo();
    }

    if (ImGui::BeginCombo("Init extractor", _extractorIdToNames.at(_iniCurrentId).c_str()))
    {
        for (int i = 0; i < _extractorIdToNames.size(); i++)
        {
            bool isSelected = (_iniCurrentId == i); // You can store your selection however you want, outside or inside your objects
            if (ImGui::Selectable(_extractorIdToNames.at(i).c_str(), isSelected))
            {
                _iniCurrentId = i;
            }
            if (isSelected)
                ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
        }
        ImGui::EndCombo();
    }

    // TODO(dgj1): display this only if a markerfile has been selected
    if (ImGui::BeginCombo("Marker extractor", _extractorIdToNames.at(_markerCurrentId).c_str()))
    {
        for (int i = 0; i < _extractorIdToNames.size(); i++)
        {
            bool isSelected = (_markerCurrentId == i); // You can store your selection however you want, outside or inside your objects
            if (ImGui::Selectable(_extractorIdToNames.at(i).c_str(), isSelected))
            {
                _markerCurrentId = i;
            }
            if (isSelected)
                ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
        }
        ImGui::EndCombo();
    }

    //if (ImGui::Button("Change features", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    //{
    //    WAIEventSetExtractors* event                  = new WAIEventSetExtractors();
    //    event->extractorIds.trackingExtractorId       = _currentId;
    //    event->extractorIds.initializationExtractorId = _iniCurrentId;
    //    event->extractorIds.markerExtractorId         = _markerCurrentId;

    //    _eventQueue->push(event);
    //}

    ImGui::End();
}
