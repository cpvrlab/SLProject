#include <imgui.h>
#include <imgui_internal.h>

#include <Utils.h>
#include <SLAverageTiming.h>
#include <SLApplication.h>
#include <CVCapture.h>
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
    if (!SLAverageTiming::instance().size())
        return;

    SLchar m[2550]; // message character array
    m[0] = 0;       // set zero length

    //sort vertically
    std::vector<SLAverageTimingBlock*> blocks;
    for (auto& block : SLAverageTiming::instance())
    {
        blocks.push_back(block.second);
    }
    std::sort(blocks.begin(), blocks.end(), [](SLAverageTimingBlock* lhs, SLAverageTimingBlock* rhs) -> bool {
        return lhs->posV < rhs->posV;
    });

    //find reference time
    SLfloat refTime = 1.0f;
    if (blocks.size())
    {
        refTime = (*blocks.begin())->val.average();
        //insert number of measurment calls
        sprintf(m + strlen(m), "Num. calls: %i\n", (SLint)(*blocks.begin())->nCalls);
    }

    //insert time measurements
    for (auto* block : blocks)
    {
        SLfloat      val   = block->val.average();
        SLfloat      valPC = Utils::clamp(val / refTime * 100.0f, 0.0f, 100.0f);
        string       name  = block->name;
        stringstream ss;
        for (int i = 0; i < block->posH; ++i)
            ss << " ";
        ss << "%s: %4.1f ms (%3d%%)\n";
        sprintf(m + strlen(m), ss.str().c_str(), name.c_str(), val, (SLint)valPC);
    }

    //define ui elements
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
    ImGui::Begin("Tracking Timing", _activator, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::TextUnformatted(m);
    ImGui::End();
    ImGui::PopFont();

}
