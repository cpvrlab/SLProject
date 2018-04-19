//#############################################################################
//  File:      SLImGuiMapStorage.cpp
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

#include <SLImGuiMapStorage.h>
#include <SLCVMap.h>
#include <SLCVMapNode.h>
#include <SLCVMapStorage.h>
#include <SLCVOrbVocabulary.h>
#include <SLCVMapTracking.h>

//-----------------------------------------------------------------------------
//const char* SLImGuiMapStorage::_currItem = NULL;
//int SLImGuiMapStorage::_currN = -1;
//-----------------------------------------------------------------------------
SLImGuiMapStorage::SLImGuiMapStorage(const string& name, SLCVMap* map, 
    SLCVMapNode* mapNode, SLCVKeyFrameDB* kfDB, SLCVMapTracking* tracking )
    : SLImGuiInfosDialog(name),
    _map(map),
    _mapNode(mapNode),
    _kfDB(kfDB),
    _tracking(tracking)
{
}
//-----------------------------------------------------------------------------
void SLImGuiMapStorage::buildInfos()
{
    if (!_map || !_mapNode)
        return;

    if (ImGui::Button("Save map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
        SLCVMapStorage::saveMap(SLCVMapStorage::getCurrentId(), *_map, true);
        //update key frames, because there may be new textures in file system
        _mapNode->updateKeyFrames(_map->GetAllKeyFrames());
    }

    ImGui::Separator();
    if (ImGui::Button("New map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
        //increase current id and maximum id in MapStorage
        SLCVMapStorage::newMap();
        //clear current field in combobox, until this new map is saved
        SLCVMapStorage::currItem = NULL;
        SLCVMapStorage::currN = -1;
    }

    ImGui::Separator();
    {
        if (ImGui::BeginCombo("Current", SLCVMapStorage::currItem)) // The second parameter is the label previewed before opening the combo.
        {
            for (int n = 0; n < SLCVMapStorage::existingMapNames.size(); n++)
            {
                bool isSelected = (SLCVMapStorage::currItem == SLCVMapStorage::existingMapNames[n].c_str()); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(SLCVMapStorage::existingMapNames[n].c_str(), isSelected)) {
                    SLCVMapStorage::currItem = SLCVMapStorage::existingMapNames[n].c_str();
                    SLCVMapStorage::currN = n;
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus();   // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }
    }

    if (ImGui::Button("Load map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
        if (SLCVMapStorage::currItem)
        {
            //reset first, otherwise loaded map will be cleared again.
            _tracking->Reset();
            //load selected map
            string selectedMapName = SLCVMapStorage::existingMapNames[SLCVMapStorage::currN];
            SLCVMapStorage storage(SLCVOrbVocabulary::get());
            storage.loadMap(selectedMapName, *_map, *_kfDB);

            //set state to initialized and lost
            _tracking->mState = SLCVMapTracking::eTrackingState::LOST;
            //update scene
            _mapNode->updateAll(*_map);
        }
    }
}
