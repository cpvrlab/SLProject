#include <imgui.h>
#include <imgui_internal.h>

#include <Utils.h>
#include <AverageTiming.h>
#include <SLApplication.h>
#include <AppDemoGuiStatsDebugTiming.h>
#include <Utils.h>
//-----------------------------------------------------------------------------
AppDemoGuiStatsDebugTiming::AppDemoGuiStatsDebugTiming(string name, bool* activator)
  : AppDemoGuiInfosDialog(name, activator)
{
}

//-----------------------------------------------------------------------------
void AppDemoGuiStatsDebugTiming::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::Begin(_name.c_str(), _activator, _initMinDialogSize);

    if (AverageTiming::instance().size())
    {
        SLchar m[2550]; // message character array
        m[0] = 0;       // set zero length

        AverageTiming::getTimingMessage(m);

        //define ui elements
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
        ImGui::TextUnformatted(m);
        ImGui::PopFont();
    }

    ImGui::End();
}
