//#############################################################################
//  File:      SLImGuiInfosMapNodeTransform.cpp
//  Author:    Michael Goettlicher, Jan Dellsperger
//  Date:      September 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <SLCVMapNode.h>
#include <SLCVMapStorage.h>
#include <SLCVMapTracking.h>
#include <SLImGuiInfosMapNodeTransform.h>

//-----------------------------------------------------------------------------
SLImGuiInfosMapNodeTransform::SLImGuiInfosMapNodeTransform(
  std::string      name,
  SLCVMapNode*     mapNode,
  SLCVMapTracking* tracking)
  : SLImGuiInfosDialog(name),
    _mapNode(mapNode),
    _tracking(tracking)
{
}
//-----------------------------------------------------------------------------
void SLImGuiInfosMapNodeTransform::buildInfos()
{
    static SLTransformSpace tSpace = TS_object;
    //rotation
    ImGui::InputFloat("Rot. Value", &_transformationRotValue, 1.0f);
    _transformationRotValue = ImClamp(_transformationRotValue, -360.0f, 360.0f);

    static SLfloat sp = 3; //spacing
    SLfloat        bW = (ImGui::GetContentRegionAvailWidth() - 2 * sp) / 3;
    if (ImGui::Button("RotX", ImVec2(bW, 0.0f)))
    {
        _mapNode->rotate(_transformationRotValue, 1, 0, 0, tSpace);
    }
    ImGui::SameLine(0.0, sp);
    if (ImGui::Button("RotY", ImVec2(bW, 0.0f)))
    {
        _mapNode->rotate(_transformationRotValue, 0, 1, 0, tSpace);
    }
    ImGui::SameLine(0.0, sp);
    if (ImGui::Button("RotZ", ImVec2(bW, 0.0f)))
    {
        _mapNode->rotate(_transformationRotValue, 0, 0, 1, tSpace);
    }
    ImGui::Separator();

    //translation
    ImGui::InputFloat("Transl. Value", &_transformationTransValue, 0.1f);

    if (ImGui::Button("TransX", ImVec2(bW, 0.0f)))
    {
        _mapNode->translate(_transformationTransValue, 0, 0, tSpace);
    }
    ImGui::SameLine(0.0, sp);
    if (ImGui::Button("TransY", ImVec2(bW, 0.0f)))
    {
        _mapNode->translate(0, _transformationTransValue, 0, tSpace);
    }
    ImGui::SameLine(0.0, sp);
    if (ImGui::Button("TransZ", ImVec2(bW, 0.0f)))
    {
        _mapNode->translate(0, 0, _transformationTransValue, tSpace);
    }
    ImGui::Separator();

    //scale
    ImGui::InputFloat("Scale Value", &_transformationScaleValue, 0.1f);
    _transformationScaleValue = ImClamp(_transformationScaleValue, 0.0f, 1000.0f);

    if (ImGui::Button("Scale", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        _mapNode->scale(_transformationScaleValue);
    }
    ImGui::Separator();

    if (ImGui::Button("Save State", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        SLCVMapStorage::saveMap(SLCVMapStorage::getCurrentId(), _tracking, true);
    }
}
