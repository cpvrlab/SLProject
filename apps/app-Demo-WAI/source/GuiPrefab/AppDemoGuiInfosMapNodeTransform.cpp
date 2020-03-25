//#############################################################################
//  File:      AppDemoGuiInfosMapNodeTransform.cpp
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

#include <AppDemoGuiInfosMapNodeTransform.h>

//-----------------------------------------------------------------------------
AppDemoGuiInfosMapNodeTransform::AppDemoGuiInfosMapNodeTransform(
  std::string            name,
  bool*                  activator,
  std::queue<WAIEvent*>* eventQueue)
  : AppDemoGuiInfosDialog(name, activator),
    _eventQueue(eventQueue)
{
}

//-----------------------------------------------------------------------------
void AppDemoGuiInfosMapNodeTransform::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin("Node Transform", _activator, ImGuiWindowFlags_AlwaysAutoResize);
    static SLTransformSpace tSpace = TS_world;
    ImGui::Text("Transf. Space:");
    ImGui::SameLine();
    ImGui::BeginGroup();
    if (ImGui::RadioButton("Object", (int*)&tSpace, 0)) tSpace = TS_object;
    ImGui::SameLine();
    if (ImGui::RadioButton("World", (int*)&tSpace, 1)) tSpace = TS_world;
    ImGui::SameLine();
    if (ImGui::RadioButton("Parent", (int*)&tSpace, 2)) tSpace = TS_parent;
    ImGui::Separator();
    ImGui::EndGroup();

    static SLfloat sp = 3; //spacing
    SLfloat        bW = (ImGui::GetContentRegionAvailWidth() - 2 * sp) / 3;

    bool transformationOccured = false;

    //rotation
    SLVec3f rotation = SLVec3f(0, 0, 0);

    if (ImGui::ButtonEx("--", ImVec2(0, 0), ImGuiButtonFlags_Repeat | ImGuiButtonFlags_PressedOnClick))
        _transformationRotValue -= 10.0f;
    ImGui::SameLine();
    if (ImGui::ButtonEx("++", ImVec2(0, 0), ImGuiButtonFlags_Repeat | ImGuiButtonFlags_PressedOnClick))
        _transformationRotValue += 10.0f;
    ImGui::SameLine();
    ImGui::InputFloat("Rot. Value", &_transformationRotValue, 0.1f, 10.f, 3);
    _transformationRotValue = ImClamp(_transformationRotValue, -360.0f, 360.0f);

    if (ImGui::Button("RotX", ImVec2(bW, 0.0f)))
    {
        rotation.x = _transformationRotValue;

        transformationOccured = true;
    }
    ImGui::SameLine(0.0, sp);
    if (ImGui::Button("RotY", ImVec2(bW, 0.0f)))
    {
        rotation.y = _transformationRotValue;

        transformationOccured = true;
    }
    ImGui::SameLine(0.0, sp);
    if (ImGui::Button("RotZ", ImVec2(bW, 0.0f)))
    {
        rotation.z = _transformationRotValue;

        transformationOccured = true;
    }
    ImGui::Separator();

    //translation
    SLVec3f translation = SLVec3f(0, 0, 0);

    if (ImGui::ButtonEx("--", ImVec2(0, 0), ImGuiButtonFlags_Repeat | ImGuiButtonFlags_PressedOnClick))
        _transformationTransValue -= 1.0f;
    ImGui::SameLine();
    if (ImGui::ButtonEx("++", ImVec2(0, 0), ImGuiButtonFlags_Repeat | ImGuiButtonFlags_PressedOnClick))
        _transformationTransValue += 1.0f;
    ImGui::SameLine();
    ImGui::InputFloat("Transl. Value", &_transformationTransValue, 0.01f, 0.1f, 3);

    if (ImGui::Button("TransX", ImVec2(bW, 0.0f)))
    {
        translation.x = _transformationTransValue;

        transformationOccured = true;
    }
    ImGui::SameLine(0.0, sp);
    if (ImGui::Button("TransY", ImVec2(bW, 0.0f)))
    {
        translation.y = _transformationTransValue;

        transformationOccured = true;
    }
    ImGui::SameLine(0.0, sp);
    if (ImGui::Button("TransZ", ImVec2(bW, 0.0f)))
    {
        translation.z = _transformationTransValue;

        transformationOccured = true;
    }
    ImGui::Separator();

    //scale
    float scale = 1.0f;

    if (ImGui::ButtonEx("--", ImVec2(0, 0), ImGuiButtonFlags_Repeat | ImGuiButtonFlags_PressedOnClick))
        _transformationScaleValue -= 0.1f;
    ImGui::SameLine();
    if (ImGui::ButtonEx("++", ImVec2(0, 0), ImGuiButtonFlags_Repeat | ImGuiButtonFlags_PressedOnClick))
        _transformationScaleValue += 0.1f;
    ImGui::SameLine();
    ImGui::InputFloat("Scale Value", &_transformationScaleValue, 0.01f, 1.f, 3);
    _transformationScaleValue = ImClamp(_transformationScaleValue, 0.0f, 1000.0f);
    if (ImGui::Button("Scale", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        scale                 = _transformationScaleValue;
        transformationOccured = true;
    }

    if (transformationOccured)
    {
        WAIEventMapNodeTransform* event = new WAIEventMapNodeTransform();

        event->tSpace      = tSpace;
        event->rotation    = rotation;
        event->translation = translation;
        event->scale       = scale;

        _eventQueue->push(event);
    }

    ImGui::End();
}
