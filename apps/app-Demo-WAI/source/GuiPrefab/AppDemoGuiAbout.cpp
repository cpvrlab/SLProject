#include <imgui.h>
#include <imgui_internal.h>

#include <SLApplication.h>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiAbout.h>
//-----------------------------------------------------------------------------
// cpvr logo not used yet, a bug must be resolved
AppDemoGuiAbout::AppDemoGuiAbout(std::string name, SLGLTexture* cpvrLogo, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
{
    _infoAbout = "Welcome to the WAI demo app. WAI stands for Where Am I. "
                 "In its core all is about detecting the pose (position and orientation) "
                 "of the live video camera. This task is the major problem to solve in "
                 "Augmented Reality application. WAI is developed at the Computer Science "
                 "Department of the Bern University of Applied Sciences.\n";

    //_cpvrLogo = cpvrLogo;
}

//-----------------------------------------------------------------------------
//! Centers the next ImGui window in the parent
void AppDemoGuiAbout::centerNextWindow(SLSceneView* sv, SLfloat widthPC, SLfloat heightPC)
{
    SLfloat width  = (SLfloat)sv->scrW() * widthPC;
    SLfloat height = (SLfloat)sv->scrH() * heightPC;
    ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiSetCond_Always);
    ImGui::SetNextWindowPosCenter(ImGuiSetCond_Always);
}

//-----------------------------------------------------------------------------
void AppDemoGuiAbout::buildInfos(SLScene* s, SLSceneView* sv)
{
    /*
    if (_cpvrLogo == nullptr)
    {
        // The texture resources get deleted by the SLScene destructor
        _cpvrLogo = new SLGLTexture("LogoCPVR_256L.png");
        if (_cpvrLogo != nullptr)
            _cpvrLogo->bindActive();
    }
    else
        _cpvrLogo->bindActive();
    */
    SLfloat iconSize = sv->scrW() * 0.15f;

    centerNextWindow(sv);
    ImGui::Begin("About WAI-Demo", _activator, ImGuiWindowFlags_NoResize);
    //ImGui::Image((ImTextureID)(intptr_t)_cpvrLogo->texName(), ImVec2(iconSize, iconSize), ImVec2(0, 1), ImVec2(1, 0));
    //ImGui::SameLine();
    ImGui::Text("Version: %s", SLApplication::version.c_str());
    ImGui::Separator();
    ImGui::Text("Git Branch: %s (Commit: %s)", SLApplication::gitBranch.c_str(), SLApplication::gitCommit.c_str());
    ImGui::Text("Git Date: %s", SLApplication::gitDate.c_str());
    ImGui::Separator();
    ImGui::TextWrapped("%s", _infoAbout.c_str());
    ImGui::End();
}

