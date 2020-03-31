#include <imgui.h>
#include <imgui_internal.h>

#include <Utils.h>
#include <AverageTiming.h>
#include <AppDemoGuiStatsDebugTiming.h>
#include <Utils.h>
//-----------------------------------------------------------------------------
AppDemoGuiStatsDebugTiming::AppDemoGuiStatsDebugTiming(string name, bool* activator, ImFont* font)
  : AppDemoGuiInfosDialog(name, activator, font)
{
}

//-----------------------------------------------------------------------------
void AppDemoGuiStatsDebugTiming::buildInfos(SLScene* s, SLSceneView* sv)
{
    ImGui::PushFont(_font);
    //todo imgui update
    ImGui::Begin(_name.c_str(), _activator /*, ImVec2(100.f, 50.f)*/);

    if (AverageTiming::instance().size())
    {
        SLchar m[2550]; // message character array
        m[0] = 0;       // set zero length

        AverageTiming::getTimingMessage(m);

        //define ui elements
        ImGui::TextUnformatted(m);
    }

    ImGui::End();
    ImGui::PopFont();
}
