#include <imgui.h>
#include <imgui_internal.h>

#include <SLApplication.h>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiUIPrefs.h>
#include <CVCapture.h>
#include <SLGLImGui.h>
#include <imgui.h>
#include <imgui_internal.h>
//-----------------------------------------------------------------------------
AppDemoGuiUIPrefs::AppDemoGuiUIPrefs(std::string name, GUIPreferences* prefs, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
{
    _prefs = prefs;
}

//-----------------------------------------------------------------------------
void AppDemoGuiUIPrefs::buildInfos(SLScene* s, SLSceneView* sv)
{
        ImGuiWindowFlags window_flags = 0;
        window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
        ImGui::Begin("User Interface Preferences", _activator, window_flags);

        ImGui::SliderFloat("Prop. Font Size", &SLGLImGui::fontPropDots, 16.f, 70.f, "%0.0f");
        ImGui::SliderFloat("Fixed Font Size", &SLGLImGui::fontFixedDots, 13.f, 50.f, "%0.0f");
        ImGuiStyle& style = ImGui::GetStyle();
        if (ImGui::SliderFloat("Item Spacing X", &style.ItemSpacing.x, 0.0f, 20.0f, "%0.0f"))
            style.WindowPadding.x = style.FramePadding.x = style.ItemInnerSpacing.x = style.ItemSpacing.x;
        if (ImGui::SliderFloat("Item Spacing Y", &style.ItemSpacing.y, 0.0f, 10.0f, "%0.0f"))
            style.WindowPadding.y = style.FramePadding.y = style.ItemInnerSpacing.y = style.ItemSpacing.y;

        ImGui::Separator();

        SLchar reset[255];

        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
        sprintf(reset, "Reset User Interface (DPI: %d)", SLApplication::dpi);
        _prefs->setDPI(SLApplication::dpi);
        if (ImGui::MenuItem(reset))
        {
            SLstring fullPathFilename = SLApplication::configPath + "DemoGui.yml";
            Utils::deleteFile(fullPathFilename);
            _prefs->load();
        }
        ImGui::PopFont();

        ImGui::End();
}
