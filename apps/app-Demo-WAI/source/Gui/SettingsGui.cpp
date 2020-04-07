#include <SettingsGui.h>
#include <ErlebAR.h>
#include <imgui_internal.h>

SettingsGui::SettingsGui(int         dotsPerInch,
                         int         screenWidthPix,
                         int         screenHeightPix,
                         std::string fontPath)
{
    resize(screenWidthPix, screenHeightPix);
}

SettingsGui::~SettingsGui()
{
}

void SettingsGui::onResize(SLint scrW, SLint scrH)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH);
}

void SettingsGui::resize(int scrW, int scrH)
{
}

void SettingsGui::build(SLScene* s, SLSceneView* sv)
{

    //ImGui::ShowMetricsWindow();
}
