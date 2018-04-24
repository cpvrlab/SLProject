//#############################################################################
//  File:      SLImGuiInfosMapTransform.cpp
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

#include <SLImGuiInfosMapTransform.h>
#include <SLCVMap.h>
#include <SLCVMapStorage.h>

//-----------------------------------------------------------------------------
SLImGuiInfosMapTransform::SLImGuiInfosMapTransform(std::string name, SLCVMap* map)
    : SLImGuiInfosDialog(name),
    _map(map)
{
}
//-----------------------------------------------------------------------------
void SLImGuiInfosMapTransform::buildInfos()
{
    //rotation
    ImGui::InputFloat("Rot. Value", &_transformationRotValue, 1.0f);
    _transformationRotValue = ImClamp(_transformationRotValue, -360.0f, 360.0f);

    static SLfloat sp = 3; //spacing
    SLfloat bW = (ImGui::GetContentRegionAvailWidth() - 2 * sp) / 3;
    if (ImGui::Button("RotX", ImVec2(bW, 0.0f))) {
        _map->applyTransformation(_transformationRotValue, SLCVMap::ROT_X);
    } ImGui::SameLine(0.0, sp);
    if (ImGui::Button("RotY", ImVec2(bW, 0.0f))) {
        _map->applyTransformation(_transformationRotValue, SLCVMap::ROT_Y);
    } ImGui::SameLine(0.0, sp);
    if (ImGui::Button("RotZ", ImVec2(bW, 0.0f))) {
        _map->applyTransformation(_transformationRotValue, SLCVMap::ROT_Z);
    }
    ImGui::Separator();

    //translation
    ImGui::InputFloat("Transl. Value", &_transformationTransValue, 0.1f);

    if (ImGui::Button("TransX", ImVec2(bW, 0.0f))) {
        _map->applyTransformation(_transformationTransValue, SLCVMap::TRANS_X);
    } ImGui::SameLine(0.0, sp);
    if (ImGui::Button("TransY", ImVec2(bW, 0.0f))) {
        _map->applyTransformation(_transformationTransValue, SLCVMap::TRANS_Y);
    } ImGui::SameLine(0.0, sp);
    if (ImGui::Button("TransZ", ImVec2(bW, 0.0f))) {
        _map->applyTransformation(_transformationTransValue, SLCVMap::TRANS_Z);
    }
    ImGui::Separator();

    //scale
    ImGui::InputFloat("Scale Value", &_transformationScaleValue, 0.1f);
    _transformationScaleValue = ImClamp(_transformationScaleValue, 0.0f, 1000.0f);

    if (ImGui::Button("Scale", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
        _map->applyTransformation(_transformationScaleValue, SLCVMap::SCALE);
    }
    ImGui::Separator();

    if (ImGui::Button("Save State", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
        SLCVMapStorage::saveMap(SLCVMapStorage::getCurrentId(), *_map, true);
    }
}