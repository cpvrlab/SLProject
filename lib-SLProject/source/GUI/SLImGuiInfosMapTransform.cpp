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
#include <SLGLImGui.h>

//-----------------------------------------------------------------------------
SLImGuiInfosMapTransform::SLImGuiInfosMapTransform(std::string name, SLCVMap* map)
    : SLImGuiInfosDialog(name),
    _map(map)
{
}
//-----------------------------------------------------------------------------
void SLImGuiInfosMapTransform::buildInfos()
{
    //-------------------------------------------------------------------------
    if (ImGui::CollapsingHeader("Alignment"))
    {
        //rotation
        ImGui::InputFloat("Rot. Value", &SLGLImGui::transformationRotValue, 0.1f);
        SLGLImGui::transformationRotValue = ImClamp(SLGLImGui::transformationRotValue, -360.0f, 360.0f);

        static SLfloat sp = 3; //spacing
        SLfloat bW = (ImGui::GetContentRegionAvailWidth() - 2 * sp) / 3;
        if (ImGui::Button("RotX", ImVec2(bW, 0.0f))) {
            _map->applyTransformation(SLGLImGui::transformationRotValue, SLCVMap::ROT_X);
        } ImGui::SameLine(0.0, sp);
        if (ImGui::Button("RotY", ImVec2(bW, 0.0f))) {
            _map->applyTransformation(SLGLImGui::transformationRotValue, SLCVMap::ROT_Y);
        } ImGui::SameLine(0.0, sp);
        if (ImGui::Button("RotZ", ImVec2(bW, 0.0f))) {
            _map->applyTransformation(SLGLImGui::transformationRotValue, SLCVMap::ROT_Z);
        }
        ImGui::Separator();

        //translation
        ImGui::InputFloat("Transl. Value", &SLGLImGui::transformationTransValue, 0.1f);

        if (ImGui::Button("TransX", ImVec2(bW, 0.0f))) {
            _map->applyTransformation(SLGLImGui::transformationTransValue, SLCVMap::TRANS_X);
        } ImGui::SameLine(0.0, sp);
        if (ImGui::Button("TransY", ImVec2(bW, 0.0f))) {
            _map->applyTransformation(SLGLImGui::transformationTransValue, SLCVMap::TRANS_Y);
        } ImGui::SameLine(0.0, sp);
        if (ImGui::Button("TransZ", ImVec2(bW, 0.0f))) {
            _map->applyTransformation(SLGLImGui::transformationTransValue, SLCVMap::TRANS_Z);
        }
        ImGui::Separator();

        //scale
        ImGui::InputFloat("Scale Value", &SLGLImGui::transformationScaleValue, 0.1f);
        SLGLImGui::transformationScaleValue = ImClamp(SLGLImGui::transformationScaleValue, 0.0f, 1000.0f);

        if (ImGui::Button("Scale", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
            _map->applyTransformation(SLGLImGui::transformationScaleValue, SLCVMap::SCALE);
        }
        ImGui::Separator();

        if (ImGui::Button("Save State", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f))) {
            _map->saveState();
        }
    }
}